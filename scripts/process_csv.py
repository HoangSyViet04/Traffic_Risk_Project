import pandas as pd
import os
import math

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn file gốc bạn vừa tải lên
INPUT_CSV = r"D:\Learn\paper_traffic\data\train_labels.csv" 
# Đường dẫn file kết quả sẽ tạo ra
OUTPUT_CSV = r"D:\Learn\paper_traffic\data\processed_train.csv"

def extract_video_id(url):
    """
    Lấy ID video từ URL.
    Ví dụ: https://.../06d501fd-a9ffc960.mov -> 06d501fd-a9ffc960
    """
    if not isinstance(url, str):
        return None
    basename = os.path.basename(url) # Lấy phần cuối: 06d501fd-a9ffc960.mov
    video_id = os.path.splitext(basename)[0] # Bỏ đuôi .mov
    return video_id

def main():
    print(f"Đang đọc file: {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {INPUT_CSV}. Hãy kiểm tra lại đường dẫn!")
        return

    data_rows = []
    
    # Duyệt qua từng hàng (mỗi hàng là 1 video với nhiều sự kiện)
    for index, row in df.iterrows():
        # 1. Lấy Video ID
        video_url = row.get('Input.Video')
        video_id = extract_video_id(video_url)
        
        if not video_id:
            continue
            
        # 2. Duyệt qua 15 sự kiện tiềm năng (Answer.1 -> Answer.15)
        for i in range(1, 16): # Từ 1 đến 15
            start_col = f'Answer.{i}start'
            end_col   = f'Answer.{i}end'
            act_col   = f'Answer.{i}action' # Đây chính là caption chính
            just_col  = f'Answer.{i}justification'
            # Kiểm tra xem cột này có tồn tại trong file không
            if start_col not in df.columns:
                break
                
            # Lấy giá trị
            start = row[start_col]
            end   = row[end_col]
            action = row[act_col]
            justification = row.get(just_col, "")
            # Kiểm tra dữ liệu rác (NaN / rỗng)
            # Nếu start là NaN thì nghĩa là hết sự kiện cho video này rồi
            if pd.isna(start) or pd.isna(end) or pd.isna(action):
                continue
            act_text = str(action).strip()
            just_text = str(justification).strip() if pd.notna(justification) else ""  
            full_caption = f"{act_text} {just_text}".strip()  
            # Lưu vào danh sách
            data_rows.append({
                'video_id': video_id,
                'start': float(start),
                'end': float(end),
                'caption': full_caption
            })

    # Tạo DataFrame kết quả
    result_df = pd.DataFrame(data_rows)
    
    # Lưu ra CSV mới
    # folder data có thể chưa tồn tại nếu bạn chưa tạo, code này sẽ tạo giúp
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    result_df.to_csv(OUTPUT_CSV, index=False)
    print("-" * 30)
    print(f"XỬ LÝ THÀNH CÔNG!")
    print(f"Đã chuyển đổi {len(df)} dòng video gốc thành {len(result_df)} mẫu training.")
    print(f"File kết quả lưu tại: {OUTPUT_CSV}")
    print("-" * 30)
    print("5 dòng đầu tiên của file mới:")
    print(result_df.head())

if __name__ == "__main__":
    main()