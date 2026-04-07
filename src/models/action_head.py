import torch
import torch.nn as nn


class ActionRegressor(nn.Module):
    """
    Action Regressor v2: MLP thuần để dự đoán 5 bước tương lai,
    mỗi bước gồm 2 giá trị (Speed, Course) = 10 output.

    Luồng Tensor:
        context_vector: [B, 512]
            -> FC1 + ReLU + Dropout: [B, 256]
            -> FC2 + ReLU + Dropout: [B, 128]
            -> FC3: [B, 10]
    """

    def __init__(self, hidden_size=512, future_steps=5, output_dim=2):
        super(ActionRegressor, self).__init__()
        self.future_steps = future_steps
        self.output_dim = output_dim

        total_output = future_steps * output_dim  # 5 * 2 = 10

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, total_output),
        )

    def forward(self, context_vector):
        """
        Args:
            context_vector: [Batch, 512]
        Returns:
            future_flat: [Batch, 10]
        """
        return self.mlp(context_vector)

    def reshape_prediction(self, future_flat):
        """
        Reshape [Batch, 10] -> [Batch, 5, 2] để tính MSE Loss.
        """
        batch_size = future_flat.size(0)
        return future_flat.view(batch_size, self.future_steps, self.output_dim)