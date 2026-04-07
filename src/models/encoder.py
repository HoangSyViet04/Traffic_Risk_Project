import torch
import torch.nn as nn

from src.models.pretrain_cnn import build_cnn5_feature_extractor


class MultimodalEncoder(nn.Module):
    """
    Multimodal Encoder v2:
    - CNN 5 lớp (giống PretrainCNN) cho ảnh [B, 3, 90, 160]
    - Feature: [B, 64] (sau GAP + flatten)
    - Projection: 64 → 256 (mở rộng đặc trưng)
    - Early Fusion: 256 + 3 = 259
    - LSTM 2 tầng: input_size=259, hidden_size=512
    - LayerNorm trên context vector
    """

    def __init__(self, hidden_size=512, sensor_dim=3, freeze_cnn=False):
        super(MultimodalEncoder, self).__init__()

        # --- NHÁNH HÌNH ẢNH (CNN Feature Extractor) ---
        self.cnn = build_cnn5_feature_extractor()
        self.freeze_cnn = freeze_cnn

        # --- PROJECTION: MỞ RỘNG ĐẶC TRƯNG ẢNH ---
        self.image_feature_dim = 64
        self.projection_dim = 256
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_feature_dim, self.projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # --- EARLY FUSION LSTM ---
        fusion_input_dim = self.projection_dim + sensor_dim  # 256 + 3 = 259
        self.lstm = nn.LSTM(
            input_size=fusion_input_dim,   # 259
            hidden_size=hidden_size,       # 512
            num_layers=2,
            batch_first=True,
            dropout=0.3,                   # dropout giữa các tầng LSTM
        )

        # LayerNorm ổn định context vector
        self.context_norm = nn.LayerNorm(hidden_size)

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
            lstm_out:       [Batch, 16, 512]
            context_vector: [Batch, 512]
        """
        batch_size, frames, C, H, W = images.shape

        # --- A. TRÍCH XUẤT ĐẶC TRƯNG ẢNH ---
        c_in = images.view(batch_size * frames, C, H, W)  # [B*16, 3, 90, 160]

        if self.freeze_cnn:
            with torch.no_grad():
                features = self.cnn(c_in)  # [B*16, 64, 1, 1]
        else:
            features = self.cnn(c_in)      # [B*16, 64, 1, 1]

        features = features.view(features.size(0), -1)           # [B*16, 64]

        # --- B. PROJECTION: MỞ RỘNG ĐẶC TRƯNG ---
        features = self.image_projection(features)               # [B*16, 256]
        features = features.view(batch_size, frames, -1)         # [B, 16, 256]

        # --- C. EARLY FUSION: NỐI IMAGE + SENSOR ---
        fused = torch.cat((features, sensors), dim=2)            # [B, 16, 259]

        # --- D. LSTM 2 TẦNG ---
        lstm_out, (h_n, c_n) = self.lstm(fused)   # lstm_out: [B, 16, 512]

        # Lấy hidden state cuối cùng của TẦNG THỨ 2
        context_vector = h_n[-1]                                 # [B, 512]
        context_vector = self.context_norm(context_vector)       # LayerNorm

        return lstm_out, context_vector