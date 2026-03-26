import torch
import torch.nn as nn

def build_cnn5_feature_extractor() -> nn.Sequential:
    """
    CNN 5 lớp cho ảnh [B, 3, 90, 160] - CẤU HÌNH TH4.
    Không sử dụng Batch Normalization. Thêm Global Average Pooling.
    
    Output feature map sau lớp cuối cùng:
        [B, 64, 1, 1]
    """
    return nn.Sequential(
        # Block 1: [B, 3, 90, 160] -> [B, 16, 45, 80]
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 2: [B, 16, 45, 80] -> [B, 32, 23, 40]
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 3: [B, 32, 23, 40] -> [B, 48, 12, 20]
        nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        # Block 4: [B, 48, 12, 20] -> [B, 64, 12, 20]
        nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        # Block 5: [B, 64, 12, 20] -> [B, 64, 12, 20]
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        # TH4: GLOBAL AVERAGE POOLING 2D
        # Ép không gian về 1x1. Output shape: [B, 64, 1, 1]
        nn.AdaptiveAvgPool2d((1, 1))
    )


class PretrainCNN(nn.Module):
    """
    Mạng pre-train TH4 (Không BN, GAP, FC hạ cánh từ 64).
    
    Input:
        images: [B, 3, 90, 160]
    Output:
        pred:   [B, 2] (Speed, Course)
    """

    def __init__(self):
        super().__init__()
        self.features = build_cnn5_feature_extractor()
        
        # --- REGRESSOR TH4 ---
        # Không dùng BatchNorm. Down dần: 64 -> 32 -> 16 -> 2
        self.regressor = nn.Sequential(
            nn.Flatten(), # Đập hộp [B, 64, 1, 1] thành [B, 64]
            
            # Hạ cánh 64 -> 32
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), # Dropout nhẹ chống overfitting

            # Hạ cánh 32 -> 16
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),

            # Output Tốc độ & Góc lái
            nn.Linear(16, 2)
        )

    def extract_flat_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 90, 160]
        Returns:
            flat: [B, 64] (Thay vì 15360 như trước đây)
        """
        feat = self.features(images)          # [B, 64, 1, 1]
        flat = feat.flatten(start_dim=1)      # [B, 64]
        return flat

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = self.extract_flat_features(images)  # [B, 64]
        pred = self.regressor(flat)                # [B, 2]
        return pred