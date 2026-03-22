import json
import re
import torch

class CustomTokenizer:
    """
    Tokenizer tự build để thay thế BERT, chuẩn hóa theo bộ từ điển 1199 từ.
    Giả lập API của HuggingFace để tương thích với Dataset và Model hiện tại.
    """
    def __init__(self, vocab_path="custom_vocab.json", max_len=30):
        # Đọc bộ từ điển bạn vừa sinh ra
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        
        # Tạo từ điển ngược để dịch từ ID ra chữ
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.max_len = max_len

        # Mapping Special Tokens (Đặt tên giống BERT để không phải sửa code cũ)
        self.pad_token_id = self.word2idx.get("<PAD>", 0)
        self.unk_token_id = self.word2idx.get("<UNK>", 1)
        self.cls_token_id = self.word2idx.get("<BOS>", 2)  # Đóng vai trò là thẻ bắt đầu
        self.sep_token_id = self.word2idx.get("<EOS>", 3)  # Đóng vai trò là thẻ kết thúc

    def __len__(self):
        return len(self.word2idx)

    def clean_text(self, text):
        """Làm sạch văn bản y hệt lúc build vocab"""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def __call__(self, text, padding='max_length', truncation=True, max_length=None, return_tensors='pt'):
        """Giả lập hàm mã hóa của AutoTokenizer"""
        length = max_length if max_length is not None else self.max_len
        
        cleaned = self.clean_text(text)
        tokens = cleaned.split()

        # Chuyển chữ thành số ID (nếu không có trong từ điển thì thành <UNK>)
        token_ids = [self.word2idx.get(word, self.unk_token_id) for word in tokens]

        # Cắt bớt nếu câu quá dài (trừ hao 2 vị trí cho BOS và EOS)
        if truncation and len(token_ids) > length - 2:
            token_ids = token_ids[:length - 2]

        # Bọc BOS và EOS vào hai đầu
        token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]

        # Padding cho đủ độ dài
        if padding == 'max_length':
            pad_len = length - len(token_ids)
            if pad_len > 0:
                token_ids = token_ids + [self.pad_token_id] * pad_len

        # Trả về format giống hệt HuggingFace
        if return_tensors == 'pt':
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        return {"input_ids": token_ids}

    def decode(self, token_ids, skip_special_tokens=True):
        """Dịch mảng ID ngược lại thành câu văn"""
        if isinstance(token_ids, torch.Tensor):
            # Xử lý nếu đưa vào tensor nhiều chiều
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze().tolist()
            else:
                token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        
        words = []
        for idx in token_ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.unk_token_id, self.cls_token_id, self.sep_token_id]:
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
            
        return " ".join(words).strip()