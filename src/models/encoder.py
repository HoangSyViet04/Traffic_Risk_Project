import torch
import torch.nn as nn

import math

from src.models.pretrain_cnn import build_cnn5_feature_extractor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence features (batch_first)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultimodalEncoder(nn.Module):
    """
    Multimodal Encoder theo hướng paper:
    - CNN 5 lớp (giống PretrainCNN) cho ảnh [B, 3, 90, 160]
    - Feature map [B, 64, 12, 20] -> flatten [B, 15360]
    - Early Fusion với sensor (3-d): 60@20x12 = 15360 + 3 = 15363
    - LSTM 2 tầng: input_size=15363, hidden_size=1024
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        sensor_dim: int = 3,
        freeze_cnn: bool = False,
        encoder_type: str = "transformer",
        transformer_nhead: int = 8,
        transformer_num_layers: int = 2,
        transformer_ff_dim: int = 2048,
        transformer_dropout: float = 0.1,
        transformer_use_cls_token: bool = True,
    ):
        super(MultimodalEncoder, self).__init__()

        # --- NHÁNH HÌNH ẢNH (CNN Feature Extractor) ---
        # Dùng đúng CNN 5 lớp như lúc pre-train.
        self.cnn = build_cnn5_feature_extractor()
        self.freeze_cnn = freeze_cnn

        self.encoder_type = str(encoder_type).lower().strip()

        # --- EARLY FUSION ---
        # CNN uses GAP -> flat feature dim = 64
        self.image_feature_dim = 64
        fusion_input_dim = self.image_feature_dim + sensor_dim
        self.fusion_input_dim = fusion_input_dim

        if self.encoder_type == "transformer":
            if hidden_size % transformer_nhead != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by transformer_nhead ({transformer_nhead})."
                )

            self.input_projection = nn.Linear(fusion_input_dim, hidden_size)
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=transformer_dropout, max_len=200)
            self.use_cls_token = bool(transformer_use_cls_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if self.use_cls_token else None

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_nhead,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

            self.rnn = None
        elif self.encoder_type == "bilstm":
            if hidden_size % 2 != 0:
                raise ValueError("For bilstm, hidden_size must be even to keep output dim stable.")
            self.rnn = nn.LSTM(
                input_size=fusion_input_dim,
                hidden_size=hidden_size // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.transformer_encoder = None
            self.input_projection = None
            self.pos_encoder = None
            self.use_cls_token = False
            self.cls_token = None
        elif self.encoder_type == "gru":
            self.rnn = nn.GRU(
                input_size=fusion_input_dim,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
            )
            self.transformer_encoder = None
            self.input_projection = None
            self.pos_encoder = None
            self.use_cls_token = False
            self.cls_token = None
        elif self.encoder_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=fusion_input_dim,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
            )
            self.transformer_encoder = None
            self.input_projection = None
            self.pos_encoder = None
            self.use_cls_token = False
            self.cls_token = None
        else:
            raise ValueError(
                "Unsupported encoder_type. Use one of: 'transformer', 'lstm', 'bilstm', 'gru'. "
                f"Got: {encoder_type!r}"
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

        features = features.flatten(start_dim=1)                 # shape: [B*16, 64]
        features = features.view(batch_size, frames, -1)         # shape: [B, 16, 64]

        # --- B. EARLY FUSION: NỐI IMAGE + SENSOR ---
        # sensors: [B, 16, 3]
        fused = torch.cat((features, sensors), dim=2)            # shape: [B, 16, 15363]

        # --- C. SEQUENCE ENCODER ---
        if self.encoder_type == "transformer":
            x = self.input_projection(fused)  # [B, T, H]
            if self.use_cls_token:
                cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, H]
                x = torch.cat([cls, x], dim=1)  # [B, 1+T, H]
            x = self.pos_encoder(x)
            out = self.transformer_encoder(x)  # [B, 1+T, H] or [B, T, H]

            if self.use_cls_token:
                context_vector = out[:, 0, :]   # [B, H]
                seq_outputs = out[:, 1:, :]     # [B, T, H]
            else:
                seq_outputs = out               # [B, T, H]
                context_vector = out.mean(dim=1)

            return seq_outputs, context_vector

        if self.encoder_type == "gru":
            seq_outputs, h_n = self.rnn(fused)
            context_vector = h_n[-1]
            return seq_outputs, context_vector

        # LSTM / BiLSTM
        seq_outputs, (h_n, _) = self.rnn(fused)
        if self.encoder_type == "bilstm":
            # h_n: [num_layers*2, B, H/2] -> take last layer's fwd & bwd
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            context_vector = torch.cat([h_fwd, h_bwd], dim=1)  # [B, H]
        else:
            context_vector = h_n[-1]
        return seq_outputs, context_vector