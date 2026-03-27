import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thay đổi đường dẫn này trỏ tới thư mục ảnh 100GB của bạn
SOURCE_DIR = "D:\\Traffic_Risk_Project\\data\\images" 

# Thư mục mới sẽ chứa ảnh đã nén (Code sẽ tự tạo nếu chưa có)
DEST_DIR = "D:\\Traffic_Risk_Project\\data\\images_resized" 

# Kích thước chuẩn của bài báo: Rộng 160, Cao 90
TARGET_WIDTH = 160
TARGET_HEIGHT = 90

def process_image(task):
    """Hàm xử lý đơn lẻ cho từng ảnh"""
    src_path, dst_path = task
    try:
        # Tạo thư mục con tương ứng (ví dụ: data/images_resized/video_001)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(src_path)
        
        if img is not None:
            # 1. Resize ảnh về 160x90 bằng thuật toán INTER_AREA (chống vỡ hạt cực tốt khi thu nhỏ)
            resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # 2. Lưu ảnh mới với chất lượng nén JPEG là 95% (Mắt thường không phân biệt được, nhưng dung lượng giảm một nửa)
            cv2.imwrite(dst_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    except Exception as e:
        pass # Bỏ qua nếu có file lỗi (không phải file ảnh)

def main():
    print("Đang quét toàn bộ dữ liệu 100GB, vui lòng đợi...")
    tasks = []
    
    # Duyệt qua tất cả các thư mục và file
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # Chỉ lấy các file ảnh
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                # Lấy đường dẫn tương đối để giữ nguyên cấu trúc thư mục video_id
                relative_path = os.path.relpath(src_path, SOURCE_DIR)
                dst_path = os.path.join(DEST_DIR, relative_path)
                
                tasks.append((src_path, dst_path))
                
    total_images = len(tasks)
    print(f"Đã tìm thấy {total_images} ảnh. Bắt đầu quá trình nén đa luồng...")
    
    # Sử dụng ThreadPoolExecutor để chạy song song nhiều ảnh cùng lúc
    # max_workers=8 (hoặc 16 tùy CPU của bạn khỏe đến đâu)
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Tích hợp thanh tiến trình tqdm để xem tiến độ chạy
        list(tqdm(executor.map(process_image, tasks), total=total_images, desc="Đang Resize"))
        
    print("\n🎉 HOÀN TẤT! Hãy kiểm tra dung lượng thư mục mới, nó sẽ nhẹ đi rất nhiều!")

if __name__ == "__main__":
    main()