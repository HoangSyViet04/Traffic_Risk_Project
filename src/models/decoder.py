import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    """
    Caption Decoder: Near-Future Aware.
    Nhận context_vector (1024) + future_flat (10) = 1034-d làm input context.
    Dùng LSTM 1 tầng (hidden=1024) với Teacher Forcing để sinh caption.
    """

    def __init__(self, context_dim, hidden_size, vocab_size, embed_size=256):
        """
        Args:
            context_dim: Kích thước vector ngữ cảnh đầu vào (1034 = 1024 + 10)
            hidden_size: Kích thước hidden state LSTM (1024)
            vocab_size:  Kích thước bộ từ điển (BERT ≈ 30522)
            embed_size:  Kích thước word embedding (256)
        """
        super(CaptionDecoder, self).__init__()

        # 1. Word Embedding: token ID -> vector
        self.embed = nn.Embedding(vocab_size, embed_size)

        # 2. Chiếu context (1034-d) về embed_size (256-d) để concat với word embeddings
        self.context_projection = nn.Linear(context_dim, embed_size)

        # 3. LSTM sinh từ (1 tầng, hidden=1024)
        self.lstm = nn.LSTM(
            input_size=embed_size,   # 256
            hidden_size=hidden_size, # 1024
            num_layers=1,
            batch_first=True
        )

        # 4. Output: hidden state -> xác suất từng từ trong vocab
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, context, captions):
        """
        Args:
            context:  [Batch, 1034]  (context_vector + future_flat đã nối)
            captions: [Batch, MaxLen]  (Token IDs của caption target)
        Returns:
            outputs: [Batch, SeqLen, vocab_size]
        """
        # Teacher Forcing: bỏ từ CUỐI, dự đoán từ TIẾP THEO
        # Input:  [CLS] The car is ...
        # Target: The car is ... [SEP]
        embeddings = self.embed(captions[:, :-1])                # shape: [B, SeqLen-1, 256]

        # Chiếu context về embed_size
        context_proj = self.context_projection(context)          # shape: [B, 256]
        context_proj = context_proj.unsqueeze(1)                 # shape: [B, 1, 256]

        # Nối context vào ĐẦU chuỗi: [Context, Word1, Word2, ...]
        inputs = torch.cat((context_proj, embeddings), dim=1)    # shape: [B, SeqLen, 256]

        # Chạy qua LSTM
        hiddens, _ = self.lstm(inputs)                           # shape: [B, SeqLen, 1024]

        # Tính xác suất từ
        outputs = self.linear(hiddens)                           # shape: [B, SeqLen, vocab_size]

        return outputs