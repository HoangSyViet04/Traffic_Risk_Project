import torch
import torch.nn as nn
from src.models.encoder import MultimodalEncoder
from src.models.action_head import ActionRegressor
from src.models.decoder import CaptionDecoder


class DrivingRiskModel(nn.Module):
    """
    Full Pipeline: Encoder -> ActionRegressor -> Decoder
    Multi-Task Learning: Sinh caption + Dự đoán motion cùng lúc.
    """

    def __init__(self, config, vocab_size):
        super(DrivingRiskModel, self).__init__()

        # 1. Encoder: Image + Sensor -> context_vector [B, 1024]
        self.encoder = MultimodalEncoder(
            hidden_size=config.HIDDEN_SIZE,
            sensor_dim=config.SENSOR_DIM
        )

        # 2. Action Head: context_vector [B, 1024] -> future_flat [B, 10]
        self.action_head = ActionRegressor(
            hidden_size=config.HIDDEN_SIZE,
            future_steps=config.FUTURE_STEPS
        )

        # 3. Decoder: (context + future) [B, 1034] -> vocab_outputs [B, SeqLen, vocab_size]
        context_dim = config.HIDDEN_SIZE + (config.FUTURE_STEPS * 2)  # 1024 + 10 = 1034
        self.decoder = CaptionDecoder(
            context_dim=context_dim,
            hidden_size=config.HIDDEN_SIZE,
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE
        )

    def forward(self, images, sensors, captions):
        """
        Args:
            images:  
            sensors:  
            captions: [B, MaxLen]
        Returns:
            vocab_outputs: [B, SeqLen, vocab_size]
            future_pred:   [B, 5, 2]
        """
        # BƯỚC 1: Encoder
        context = self.encoder(images, sensors)              # shape: [B, 1024]

        # BƯỚC 2: Action Regressor (MLP)
        future_flat = self.action_head(context)              # shape: [B, 10]

        # BƯỚC 3: Nối context + future -> decoder input
        decoder_context = torch.cat((context, future_flat), dim=1)  # shape: [B, 1034]

        # BƯỚC 4: Decoder sinh caption
        vocab_outputs = self.decoder(decoder_context, captions)     # shape: [B, SeqLen, vocab_size]

        # Reshape future_flat [B, 10] -> [B, 5, 2] để tính MSE Loss
        future_pred = self.action_head.reshape_prediction(future_flat)  # shape: [B, 5, 2]

        return vocab_outputs, future_pred