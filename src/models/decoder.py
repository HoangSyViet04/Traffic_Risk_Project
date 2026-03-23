import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """Cơ chế Soft-Attention để quét qua các khung hình (frames)"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(TemporalAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [B, Frames, 1024]
        # decoder_hidden:  [B, 1024]
        dec_proj = self.decoder_att(decoder_hidden).unsqueeze(1)    # [B, 1, att_dim]
        enc_proj = self.encoder_att(encoder_outputs)                # [B, Frames, att_dim]
        
        # Tính điểm số cho từng frame
        att_energy = self.full_att(torch.tanh(enc_proj + dec_proj)) # [B, Frames, 1]
        alpha = F.softmax(att_energy, dim=1)                        # [B, Frames, 1]
        
        # Nhặt đặc trưng tương ứng với trọng số Attention
        attention_context = (encoder_outputs * alpha).sum(dim=1)    # [B, 1024]
        return attention_context, alpha

class CaptionDecoder(nn.Module):
    def __init__(self, context_dim, hidden_size, vocab_size, embed_size=256):
        super(CaptionDecoder, self).__init__()
        self.vocab_size = vocab_size

        self.context_projection = nn.Linear(context_dim, embed_size)
        # 1. Attention Module
        self.attention = TemporalAttention(hidden_size, hidden_size, 512)

        # 2. Lớp khởi tạo bộ nhớ ban đầu cho LSTM từ context vector 1034-d
        self.init_h = nn.Linear(context_dim, hidden_size)
        self.init_c = nn.Linear(context_dim, hidden_size)

        self.embed = nn.Embedding(vocab_size, embed_size)

        # 3. Đổi LSTM thành LSTMCell để chạy từng bước. 
        # Input = Word Embedding (256) + Attention Context (1024)
        self.lstm_cell = nn.LSTMCell(
            input_size=embed_size + hidden_size, 
            hidden_size=hidden_size
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_outputs, context, future_flat, captions):
        B = encoder_outputs.size(0)
        seq_len = captions.size(1)

        # Trộn ngữ cảnh và tương lai để làm bệ phóng ban đầu
        decoder_context = torch.cat((context, future_flat), dim=1) # [B, 1034]
        h = self.init_h(decoder_context) # [B, 1024]
        c = self.init_c(decoder_context) # [B, 1024]
        # Chuẩn bị trước các từ nhúng
        embeddings = self.embed(captions) # [B, SeqLen, 256]
        # Tạo vector hình ảnh để mồi cho nhịp t=0
        context_proj = self.context_projection(decoder_context) # [B, 256]

        outputs = torch.zeros(B, seq_len, self.vocab_size).to(context.device)

        # Vòng lặp xé lẻ từng từ để dịch
        for t in range(seq_len):
            if t == 0:
                # Ở nhịp đầu tiên, ép nó lấy Hình Ảnh làm đầu vào, không cho chép chữ
                word_embed = context_proj
            else:
                # Ở các nhịp sau, đưa chữ của nhịp TRƯỚC (t-1) để dự đoán chữ HIỆN TẠI (t)
                word_embed = embeddings[:, t-1, :]
            
            # CẤY MẮT ATTENTION: "Nhìn" xem frame nào quan trọng
            att_context, _ = self.attention(encoder_outputs, h) # [B, 1024]
            
            # Gộp Từ Vựng + Hình ảnh lại đưa vào não (LSTM)
            lstm_input = torch.cat([word_embed, att_context], dim=1) # [B, 1280]
            
            h, c = self.lstm_cell(lstm_input, (h, c))
            outputs[:, t, :] = self.linear(h)

        return outputs

    def generate_beam_search(self, encoder_outputs, context, future_flat, start_token_id, end_token_id, max_len=30, beam_size=3):
        device = context.device
        
        # Khởi động não bộ
        decoder_context = torch.cat((context, future_flat), dim=1)
        h = self.init_h(decoder_context)
        c = self.init_c(decoder_context)

        context_proj = self.context_projection(decoder_context) 
        att_context, _ = self.attention(encoder_outputs, h)
        lstm_input = torch.cat([context_proj, att_context], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))
        
        beams = [(0.0, [start_token_id], h, c)]
        
        for step in range(max_len):
            new_beams = []
            
            for score, tokens, h_prev, c_prev in beams:
                if tokens[-1] == end_token_id:
                    new_beams.append((score, tokens, h_prev, c_prev))
                    continue
                
                last_token = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                emb = self.embed(last_token) 
                
                att_context, _ = self.attention(encoder_outputs, h_prev)
                lstm_input = torch.cat([emb, att_context], dim=1)
                
                h_next, c_next = self.lstm_cell(lstm_input, (h_prev, c_prev))
                
                logits = self.linear(h_next)
                log_probs = F.log_softmax(logits, dim=1)
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)
                
                for i in range(beam_size):
                    next_token = topk_indices[0, i].item()
                    next_score = score + topk_log_probs[0, i].item()
                    new_beams.append((next_score, tokens + [next_token], h_next, c_next))
            
            new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]
            if all(b[1][-1] == end_token_id for b in beams):
                break
                
        return beams[0][1]