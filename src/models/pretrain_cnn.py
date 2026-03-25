import torch
import torch.nn as nn


def build_cnn5_feature_extractor() -> nn.Sequential:
    """
    CNN 5 lớp cho ảnh [B, 3, 90, 160].

    Output feature map sau lớp conv thứ 5:
        [B, 512, 3, 5]
    """
    return nn.Sequential(
        # Block 1: [B, 3, 90, 160] -> [B, 32, 45, 80]
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 2: [B, 32, 45, 80] -> [B, 64, 23, 40]
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 3: [B, 64, 23, 40] -> [B, 128, 12, 20]
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 4: [B, 128, 12, 20] -> [B, 256, 6, 10]
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 5: [B, 256, 6, 10] -> [B, 512, 3, 5]
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
    )


class PretrainCNN(nn.Module):
    """
    Mạng pre-train để dự đoán trạng thái xe từ ảnh đơn.

    Input:
        images: [B, 3, 90, 160]
    Output:
        pred:   [B, 2] (Speed, Course)
    """

    def __init__(self):
        super().__init__()
        self.features = build_cnn5_feature_extractor()
        flat_dim = 512 * 3 * 5  # 7680
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(flat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def extract_flat_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 90, 160]
        Returns:
            flat: [B, 7680]
        """
        feat = self.features(images)          # [B, 512, 3, 5]
        flat = feat.flatten(start_dim=1)      # [B, 7680]
        return flat

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = self.extract_flat_features(images)  # [B, 7680]
        pred = self.regressor(flat)                # [B, 2]
        return pred
