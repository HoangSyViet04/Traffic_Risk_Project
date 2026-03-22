import pandas as pd
import json
from collections import Counter
import re
from src.config import Config
import os



def clean_text(text):
    """Làm sạch văn bản: đưa về chữ thường, bỏ dấu câu thừa."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def build_vocabulary(csv_path, min_freq=2, save_path=Config.VOCAB_PATH):
    print(f"Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Thu thập tất cả các từ trong cột caption
    all_words = []
    for caption in df['caption']:
        cleaned = clean_text(caption)
        tokens = cleaned.split()
        all_words.extend(tokens)
        
    # 2. Đếm tần suất
    word_counts = Counter(all_words)
    print(f"Tổng số từ vựng duy nhất tìm thấy: {len(word_counts)}")
    
    # 3. Lọc những từ xuất hiện >= min_freq
    # Điều chỉnh min_freq để ép số lượng từ về sát mốc 1290 của bài báo
    valid_words = [word for word, count in word_counts.items() if count >= min_freq]
    valid_words.sort() # Sắp xếp theo alphabet cho gọn gàng
    
    print(f"Số lượng từ sau khi lọc (xuất hiện >= {min_freq} lần): {len(valid_words)}")
    
    # 4. Thêm các Special Tokens theo chuẩn
    # <PAD>: Padding (để chuỗi có độ dài bằng nhau)
    # <UNK>: Unknown (những từ quá hiếm bị loại bỏ sẽ thành từ này)
    # <BOS>: Begin of Sentence
    # <EOS>: End of Sentence
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3
    }
    
    # 5. Đánh ID cho các từ còn lại
    start_idx = len(vocab)
    for idx, word in enumerate(valid_words):
        vocab[word] = start_idx + idx
        
    print(f"Kích thước bộ từ điển cuối cùng (đã cộng Special Tokens): {len(vocab)}")
    
    # 6. Lưu ra file JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
        
    print(f"Đã lưu bộ từ điển thành công tại: {save_path}")

if __name__ == "__main__":
    # Bạn có thể điều chỉnh min_freq = 1, 2, 3... để xem số lượng từ thay đổi thế nào
    build_vocabulary(Config.TRAIN_CSV, min_freq=2, save_path=Config.VOCAB_PATH)