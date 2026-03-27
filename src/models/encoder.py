import torch
import torch.nn as nn

from src.models.pretrain_cnn import build_cnn5_feature_extractor


class MultimodalEncoder(nn.Module):
    """
    Multimodal Encoder theo hướng paper:
    - CNN 5 lớp (giống PretrainCNN) cho ảnh [B, 3, 90, 160]
    - Feature map [B, 64, 12, 20] -> flatten [B, 15360]
    - Early Fusion với sensor (3-d): 60@20x12 = 15360 + 3 = 15363
    - LSTM 2 tầng: input_size=15363, hidden_size=1024
    """

    def __init__(self, hidden_size=1024, sensor_dim=3, freeze_cnn=True):
        super(MultimodalEncoder, self).__init__()

        # --- NHÁNH HÌNH ẢNH (CNN Feature Extractor) ---
        # Dùng đúng CNN 5 lớp như lúc pre-train.
        self.cnn = build_cnn5_feature_extractor()
        self.freeze_cnn = freeze_cnn

        # --- EARLY FUSION LSTM ---
        # Input size = flattened image feature (15360) + sensor (3) = 15363
        self.image_feature_dim = 64 * 12 * 20
        fusion_input_dim = self.image_feature_dim + sensor_dim
        self.lstm = nn.LSTM(
            input_size=fusion_input_dim,  # 15363
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

        missing, unexpected = self.cnn.load_state_dict(cnn_state, strict=False)
        print(f"Loaded pretrained CNN from: {path}")
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
                features = self.cnn(c_in)  # shape: [B*16, 64, 12, 20]
        else:
            features = self.cnn(c_in)      # shape: [B*16, 64, 12, 20]

        features = features.view(features.size(0), -1)           # shape: [B*16, 15360]
        features = features.view(batch_size, frames, -1)         # shape: [B, 16, 15360]

        # --- B. EARLY FUSION: NỐI IMAGE + SENSOR ---
        # sensors: [B, 16, 3]
        fused = torch.cat((features, sensors), dim=2)            # shape: [B, 16, 15363]

        # --- C. LSTM 2 TẦNG ---
        # lstm_out: [B, 16, 1024] (output tại mọi timestep)
        # h_n:      [2, B, 1024]  (hidden state cuối của mỗi tầng)
        lstm_out, (h_n, c_n) = self.lstm(fused)

        # Lấy hidden state cuối cùng của TẦNG THỨ 2 (index -1)
        context_vector = h_n[-1]                                 # shape: [B, 1024]

        return context_vector