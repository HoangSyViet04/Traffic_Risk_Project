import torch
import torch.nn as nn
from src.models.encoder import MultimodalEncoder
from src.models.action_head import ActionRegressor
from src.models.decoder import CaptionDecoder

class DrivingRiskModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(DrivingRiskModel, self).__init__()

        self.encoder = MultimodalEncoder(
            hidden_size=config.HIDDEN_SIZE,
            sensor_dim=config.SENSOR_DIM
        )

        self.action_head = ActionRegressor(
            hidden_size=config.HIDDEN_SIZE,
            future_steps=config.FUTURE_STEPS
        )

        context_dim = config.HIDDEN_SIZE + (config.FUTURE_STEPS * 2)  
        self.decoder = CaptionDecoder(
            context_dim=context_dim,
            hidden_size=config.HIDDEN_SIZE,
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE
        )

    def forward(self, images, sensors, captions):
        # BƯỚC 1: Encoder nhả ra cả chuỗi hình ảnh VÀ cục tổng kết
        encoder_outputs, context = self.encoder(images, sensors)     

        # BƯỚC 2: Action dự đoán tương lai
        future_flat = self.action_head(context)              

        # BƯỚC 3: Decoder cắm đủ 3 dây (Chuỗi ảnh, Tổng kết, Tương lai)
        vocab_outputs = self.decoder(encoder_outputs, context, future_flat, captions)     

        future_pred = self.action_head.reshape_prediction(future_flat)  

        return vocab_outputs, future_pred