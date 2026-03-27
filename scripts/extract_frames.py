import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

VIDEO_DIR = r"D:\Learn\paper_traffic\data\raw_videos"  
OUTPUT_DIR = r"D:\Learn\paper_traffic\data\images"     

def extract_frames(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Tạo folder riêng cho video này
    save_folder = os.path.join(OUTPUT_DIR, video_name)
    if os.path.exists(save_folder):
        return 
    os.makedirs(save_folder, exist_ok=True)

    # Đọc video
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 6 == 0:
            saved_count +=1
            frame_name = f"frame_{saved_count}.jpg"
            save_path = os.path.join(save_folder, frame_name)
            cv2.imwrite(save_path, frame)
        
        count += 1
    cap.release()

if __name__ == "__main__":
    # Chạy vòng lặp cho tất cả video
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mov")) 
    print(f"Tìm thấy {len(video_files)} videos. Bắt đầu cắt song song...")
    max_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(extract_frames, video_files), total=len(video_files)))

    print("Xong! Kiểm tra folder data/images")