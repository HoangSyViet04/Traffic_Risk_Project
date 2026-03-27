import os
import pandas as pd
import shutil
from tqdm import tqdm

# --- CẤU HÌNH ---
CSV_FILE = r"D:\Learn\paper_traffic\data\train_labels.csv"     # File danh sách chuẩn

# Đây là folder chứa toàn bộ 100k file JSON (File gốc lộn xộn)
SOURCE_DIR = r"D:\Learn\paper_traffic\data\telemetry"             

# Đây là folder MỚI, sẽ chỉ chứa những file sạch được copy sang
DEST_DIR = r"D:\Learn\paper_traffic\data\telemetry_filtered"      

def main():
    # 1. Đọc danh sách ID cần thiết
    print(f"Đang đọc CSV: {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        # Tạo tập hợp các ID cần lấy (dùng set để tra cứu cực nhanh)
        valid_ids = set(df['video_id'].unique())
        print(f"Cần tìm {len(valid_ids)} file JSON.")
    except Exception as e:
        print(f" Lỗi: {e}")
        return

    # 2. Tạo folder đích nếu chưa có
    if not os.path.exists(DEST_DIR):
        print(f" Đang tạo folder mới: {DEST_DIR}")
        os.makedirs(DEST_DIR)
    else:
        print(f" Folder đích đã tồn tại: {DEST_DIR}")

    # 3. Bắt đầu quá trình copy
    print("\n BẮT ĐẦU COPY...")
    
    copied_count = 0
    missing_count = 0
    
    # Duyệt qua danh sách ID trong CSV để đi tìm file
    for video_id in tqdm(valid_ids):
        # Tên file gốc
        file_name = f"{video_id}.json"
        
        # Đường dẫn nguồn và đích
        src_path = os.path.join(SOURCE_DIR, file_name)
        dest_path = os.path.join(DEST_DIR, file_name)
        
        # Kiểm tra xem file nguồn có tồn tại không
        if os.path.exists(src_path):
            # COPY FILE (shutil.copy2 giữ nguyên metadata ngày giờ tạo file)
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        else:
            # Có trong CSV mà không thấy file JSON đâu
            missing_count += 1

    # 4. Tổng kết
    print("\n" + "="*30)
    print("KẾT QUẢ")
    print("="*30)
    print(f"Đã copy thành công: {copied_count} files")
    print(f"Không tìm thấy: {missing_count} files")
    print(f"Folder sạch nằm tại: {DEST_DIR}")
    
    if missing_count > 0:
        print("Lưu ý: Một số video trong CSV không có dữ liệu JSON tương ứng.")

if __name__ == "__main__":
    main()