import torch
import torch.nn as nn

def build_cnn5_feature_extractor() -> nn.Sequential:
    """
    CNN 5 lớp cho ảnh [B, 3, 90, 160].

    Output feature map sau lớp conv thứ 5:
        [B, 64, 12, 20]
    """
    return nn.Sequential(
        # Block 1: [B, 3, 90, 160] -> [B, 16, 45, 80]
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 2: [B, 16, 45, 80] -> [B, 32, 23, 40]
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 3: [B, 32, 23, 40] -> [B, 48, 12, 20]
        nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 4: [B, 48, 12, 20] -> [B, 64, 12, 20]
        nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        # Block 5: [B, 64, 12, 20] -> [B, 64, 12, 20]
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )


class PretrainCNN(nn.Module):
    """
    Mạng pre-train để dự đoán trạng thái xe từ ảnh đơn (ĐÃ CẢI TIẾN BOTTLENECK).
    
    Input:
        images: [B, 3, 90, 160]
    Output:
        pred:   [B, 2] (Speed, Course)
    """

    def __init__(self):
        super().__init__()
        self.features = build_cnn5_feature_extractor()
        
        
        self.regressor = nn.Sequential(
            # Hạ cánh 15360 -> 1024
            nn.Linear(64 * 12 * 20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Hạ cánh 1024 -> 256
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Hạ cánh 256 -> 32
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),

            # Output Tốc độ & Góc lái
            nn.Linear(32, 2)
        )

    def extract_flat_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 90, 160]
        Returns:
            flat: [B, 15360]
        """
        feat = self.features(images)          # [B, 64, 12, 20]
        flat = feat.flatten(start_dim=1)      # [B, 15360]
        return flat

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = self.extract_flat_features(images)  # [B, 15360]
        pred = self.regressor(flat)                # [B, 2]
        return pred