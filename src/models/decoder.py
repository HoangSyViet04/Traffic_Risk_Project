import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def generate_beam_search(
        self,
        context,
        start_token_id,
        end_token_id,
        max_len=30,
        beam_size=3,
        length_penalty_alpha: float = 0.7,
        min_len: int = 3,
    ):
        """
        Sinh caption bằng Beam Search cho 1 sample (Batch Size = 1).
        Args:
            context: [1, 1034] (Vector ngữ cảnh)
            start_token_id: ID của token <BOS> (VD: tokenizer.cls_token_id)
            end_token_id: ID của token <EOS> (VD: tokenizer.sep_token_id)
            max_len: Độ dài tối đa của caption
            beam_size: Số lượng nhánh (k) muốn theo dõi
        """
        device = context.device
        
        # 1. Khởi động LSTM bằng context vector (Giống hệt bước đầu trong hàm forward)
        context_proj = self.context_projection(context)      # [1, 256]
        context_proj = context_proj.unsqueeze(1)             # [1, 1, 256]
        
        # Đưa context qua LSTM để lấy hidden state ban đầu
        _, (h, c) = self.lstm(context_proj)
        
        def _length_penalty(seq_len: int) -> float:
            # Google NMT length penalty: ((5+len)/6)^alpha
            # With log-prob sums (negative), dividing by lp reduces the short-sequence bias.
            if length_penalty_alpha <= 0:
                return 1.0
            return float(((5.0 + seq_len) / 6.0) ** length_penalty_alpha)

        # 2. Khởi tạo Beam
        # Cấu trúc 1 beam: (raw_log_prob_sum, [token_ids], h, c)
        beams = [(0.0, [start_token_id], h, c)]
        
        # 3. Vòng lặp sinh từ
        for step in range(max_len):
            new_beams = []
            
            for score, tokens, h_prev, c_prev in beams:
                # Nếu beam này đã gặp thẻ <EOS>, giữ nguyên và đưa vào danh sách mới
                if tokens[-1] == end_token_id:
                    new_beams.append((score, tokens, h_prev, c_prev))
                    continue
                
                # Lấy token cuối cùng ra để đoán token tiếp theo
                last_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                emb = self.embed(last_token)  # [1, 1, 256]
                
                # Cho qua LSTM
                out, (h_next, c_next) = self.lstm(emb, (h_prev, c_prev))
                
                # Tính xác suất cho các từ tiếp theo
                logits = self.linear(out.squeeze(1))          # [1, vocab_size]
                log_probs = F.log_softmax(logits, dim=1)      # [1, vocab_size]

                # Avoid ending too early.
                if step < max(0, min_len - 1):
                    log_probs[0, end_token_id] = -1e9
                
                # Lấy top k từ có xác suất cao nhất (k = beam_size)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)
                
                # Phân nhánh: Tạo beam mới cho mỗi từ trong top k
                for i in range(beam_size):
                    next_token = topk_indices[0, i].item()
                    next_score = score + topk_log_probs[0, i].item()
                    new_beams.append((next_score, tokens + [next_token], h_next, c_next))
            
            # Sắp xếp lại tất cả các nhánh theo điểm số (từ cao xuống thấp)
            # Sort by length-normalized score for better BLEU/CIDEr.
            new_beams = sorted(
                new_beams,
                key=lambda x: x[0] / _length_penalty(len(x[1])),
                reverse=True,
            )
            
            # Cắt tỉa: Chỉ giữ lại top 'beam_size' nhánh tốt nhất
            beams = new_beams[:beam_size]
            
            # Tối ưu: Nếu tất cả các beam tốt nhất đều đã chạm <EOS>, dừng sớm
            if all(b[1][-1] == end_token_id for b in beams):
                break
                
        # 4. Trả về chuỗi token của nhánh có điểm số cao nhất
        best_beam = max(beams, key=lambda x: x[0] / _length_penalty(len(x[1])))
        best_tokens = best_beam[1]
        
        return best_tokens