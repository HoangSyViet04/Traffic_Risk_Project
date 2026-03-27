import torch
import torch.nn as nn


class ActionRegressor(nn.Module):
    """
    Action Regressor: dự đoán 5 bước tương lai, mỗi bước gồm
    2 giá trị (Speed, Course) = 10 output.

    Luồng Tensor:
        context_vector: [B, 1024]
            -> unsqueeze(1): [B, 1, 1024]
            -> LSTM: hidden [B, 1164]
            -> FC1 + ReLU: [B, 100]
            -> FC2 + ReLU: [B, 50]
            -> FC3: [B, 10]
    """

    def __init__(self, hidden_size=1024, future_steps=5, output_dim=2):
        super(ActionRegressor, self).__init__()
        self.future_steps = future_steps
        self.output_dim = output_dim

        total_output = future_steps * output_dim  # 5 * 2 = 10

        # LSTM layer:
        # [B, 1, 1024] -> hidden state [1, B, 1164]
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=1164,
            batch_first=True,
        )

        # Fully connected stack:
        # 1164 -> 100 -> 50 -> 10
        self.fc1 = nn.Linear(1164, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, total_output)
        self.relu = nn.ReLU()

    def forward(self, context_vector):
        """
        Args:
            context_vector: [Batch, 1024]
        Returns:
            future_flat: [Batch, 10]
        """
        # [B, 1024] -> [B, 1, 1024]
        lstm_input = context_vector.unsqueeze(1)

        # lstm_out: [B, 1, 1164]
        # h_n: [1, B, 1164]
        _, (h_n, _) = self.lstm(lstm_input)

        # Lấy hidden state cuối của LSTM: [B, 1164]
        hidden = h_n[-1]

        # [B, 1164] -> [B, 100]
        hidden = self.relu(self.fc1(hidden))

        # [B, 100] -> [B, 50]
        hidden = self.relu(self.fc2(hidden))

        # [B, 50] -> [B, 10]
        future_flat = self.fc3(hidden)
        return future_flat

    def reshape_prediction(self, future_flat):
        """
        Hàm phụ trợ: Reshape [Batch, 10] -> [Batch, 5, 2] để tính MSE Loss.
        """
        batch_size = future_flat.size(0)
        # shape: [B, 10] -> [B, 5, 2]
        return future_flat.view(batch_size, self.future_steps, self.output_dim)