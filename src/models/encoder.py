import torch
import torch.nn as nn

from src.models.pretrain_cnn import build_cnn5_feature_extractor


class MultimodalEncoder(nn.Module):
    """
    Multimodal Encoder theo hướng paper:
    - CNN 5 lớp (giống PretrainCNN) cho ảnh [B, 3, 90, 160]
    - Feature map [B, 512, 3, 5] -> flatten [B, 7680]
    - Early Fusion với sensor (3-d): 7680 + 3 = 7683
    - LSTM 2 tầng: input_size=7683, hidden_size=1024
    """

    def __init__(self, hidden_size=1024, sensor_dim=3, freeze_cnn=True):
        super(MultimodalEncoder, self).__init__()

        # --- NHÁNH HÌNH ẢNH (CNN Feature Extractor) ---
        # Dùng đúng CNN 5 lớp như lúc pre-train.
        self.cnn = build_cnn5_feature_extractor()
        self.freeze_cnn = freeze_cnn

        # --- EARLY FUSION LSTM ---
        # Input size = flattened image feature (7680) + sensor (3) = 7683
        self.image_feature_dim = 512 * 3 * 5
        fusion_input_dim = self.image_feature_dim + sensor_dim
        self.lstm = nn.LSTM(
            input_size=fusion_input_dim,  # 7683
            hidden_size=hidden_size,      # 1024
            num_layers=2,                 # 2 tầng LSTM
            batch_first=True
        )

    def load_pretrained_cnn(self, path):
        """
        Load trọng số từ file pretrain (cnn_pretrained.pth) vào nhánh CNN.
        Tự động bỏ regressor head (Linear) vì encoder không dùng lớp đó.
        """
        state = torch.load(path, map_location="cpu")

        cleaned_state = {}
        for k, v in state.items():
            key = k.replace("module.", "")
            cleaned_state[key] = v

        cnn_state = {}
        for k, v in cleaned_state.items():
            if k.startswith("features."):
                # Key của PretrainCNN: features.*
                cnn_state[k[len("features."):]] = v
            elif k.startswith("cnn."):
                # Hỗ trợ key dạng cnn.* nếu có
                cnn_state[k[len("cnn."):]] = v

        current = self.cnn.state_dict()
        filtered_state = {}
        skipped_shape = []
        skipped_missing = []

        for k, v in cnn_state.items():
            if k not in current:
                skipped_missing.append(k)
                continue
            if tuple(current[k].shape) != tuple(v.shape):
                skipped_shape.append((k, tuple(v.shape), tuple(current[k].shape)))
                continue
            filtered_state[k] = v

        missing, unexpected = self.cnn.load_state_dict(filtered_state, strict=False)
        print(f"Loaded pretrained CNN from: {path}")
        if skipped_shape:
            print(f"Skipped {len(skipped_shape)} keys due to shape mismatch (likely different architecture).")
        if skipped_missing:
            print(f"Skipped {len(skipped_missing)} keys not present in current CNN.")
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)

    def forward(self, images, sensors):
        """
        Args:
            images:  [Batch, 16, 3, 90, 160]
            sensors: [Batch, 16, 3]  (speed, acceleration, course)
        Returns:
            context_vector: [Batch, 1024]
        """
        batch_size, frames, C, H, W = images.shape

        # --- A. TRÍCH XUẤT ĐẶC TRƯNG ẢNH ---
        # Gộp Batch*Frames để đưa qua CNN một lượt
        c_in = images.view(batch_size * frames, C, H, W)  # shape: [B*16, 3, 90, 160]

        if self.freeze_cnn:
            with torch.no_grad():
                features = self.cnn(c_in)  # shape: [B*16, 512, 3, 5]
        else:
            features = self.cnn(c_in)      # shape: [B*16, 512, 3, 5]

        features = features.view(features.size(0), -1)           # shape: [B*16, 7680]
        features = features.view(batch_size, frames, -1)         # shape: [B, 16, 7680]

        # --- B. EARLY FUSION: NỐI IMAGE + SENSOR ---
        # sensors: [B, 16, 3]
        fused = torch.cat((features, sensors), dim=2)            # shape: [B, 16, 7683]

        # --- C. LSTM 2 TẦNG ---
        # lstm_out: [B, 16, 1024] (output tại mọi timestep)
        # h_n:      [2, B, 1024]  (hidden state cuối của mỗi tầng)
        lstm_out, (h_n, c_n) = self.lstm(fused)

        # Lấy hidden state cuối cùng của TẦNG THỨ 2 (index -1)
        context_vector = h_n[-1]                                 # shape: [B, 1024]

        return context_vector