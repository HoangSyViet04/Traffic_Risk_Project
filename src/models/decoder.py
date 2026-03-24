import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Bơm thêm nhận thức về Vị Trí (Thời gian) cho Transformer"""
    def __init__(self, d_model, dropout=0.3, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CaptionDecoder(nn.Module):
    """Baby Transformer Decoder: Nhỏ gọn, tinh võ, chống học vẹt."""
    def __init__(self, context_dim, hidden_size, vocab_size, embed_size=256, num_layers=2, nhead=4, dropout=0.3):
        super(CaptionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # 1. Chiếu các đặc trưng về cùng 256-d
        self.context_projection = nn.Linear(context_dim, embed_size)      
        self.memory_projection = nn.Linear(hidden_size, embed_size)       

        # 2. Xử lý Ngôn Ngữ
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout=dropout)

        # 3. Lõi Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=512,  
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """Mặt nạ che tương lai, ép Transformer phải học đàng hoàng"""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, encoder_outputs, context, future_flat, captions):
        device = captions.device
        seq_len = captions.size(1)

        # --- CHUẨN BỊ BỘ NHỚ (MEMORY) ---
        memory = self.memory_projection(encoder_outputs)  # [B, Frames, 256]

        # --- ĐOẠN VÁ LỖI BƯỚC 1 (TRÁNH COPY LÚC TRAIN) ---
        decoder_context = torch.cat((context, future_flat), dim=1)
        
        # Tạo "Mồi nhử" bằng vector Tổng kết Hình Ảnh
        ctx_proj = self.context_projection(decoder_context).unsqueeze(1) # [B, 1, 256]

        # Cắt bỏ chữ cuối cùng, lùi chuỗi lại 1 nhịp 
        embeddings = self.embed(captions[:, :-1]) * math.sqrt(self.embed_size) # [B, SeqLen-1, 256]
        
        # Nối "Mồi Hình Ảnh" vào đầu chuỗi: [Hình Ảnh, Từ 1, Từ 2, ...] 
        tgt_emb = torch.cat([ctx_proj, embeddings], dim=1) # [B, SeqLen, 256]
        # ------------------------------------------------

        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)

        # --- DỊCH THUẬT ---
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.linear(output) # [B, SeqLen, vocab_size]
        return logits

    def generate_beam_search(self, encoder_outputs, context, future_flat, start_token_id, end_token_id, max_len=30, beam_size=3):
        device = context.device
        
        memory = self.memory_projection(encoder_outputs) # [1, Frames, 256]
        
        decoder_context = torch.cat((context, future_flat), dim=1)
        ctx_proj = self.context_projection(decoder_context).unsqueeze(1) # [1, 1, 256]
        
        beams = [(0.0, [start_token_id])]
        
        for step in range(max_len):
            new_beams = []
            
            for score, tokens in beams:
                if tokens[-1] == end_token_id:
                    new_beams.append((score, tokens))
                    continue
                
                # Biến đổi chuỗi từ đang có
                tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=device) # [1, L]
                tgt_emb_words = self.embed(tgt_tensor) * math.sqrt(self.embed_size)
                
                # --- ĐOẠN VÁ LỖI BƯỚC 2 (TRÁNH MÙ LÚC TEST) ---
                # Nối Mồi Hình Ảnh vào trước chuỗi từ [Hình Ảnh, BOS, Từ 1, ...]
                tgt_emb = torch.cat([ctx_proj, tgt_emb_words], dim=1) # [1, 1+L, 256]
                # ----------------------------------------------

                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1), device)
                
                # Phóng Transformer
                output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # Chỉ lấy vị trí CUỐI CÙNG để dự đoán từ tiếp theo
                last_out = output[:, -1, :] 
                logits = self.linear(last_out)
                log_probs = F.log_softmax(logits, dim=1)
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)
                
                for i in range(beam_size):
                    next_token = topk_indices[0, i].item()
                    next_score = score + topk_log_probs[0, i].item()
                    new_beams.append((next_score, tokens + [next_token]))
            
            new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]
            
            if all(b[1][-1] == end_token_id for b in beams):
                break
                
        return beams[0][1]