import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from PIL import Image
import numpy as np

class DrivingRiskDataset(Dataset):
    def __init__(
        self,
        csv_file,
        images_root,
        telemetry_root,
        tokenizer,
        transform=None,
        max_frames=16,
        future_steps=5,
        sample_fps: int = 1,
        source_fps: int = 5,
    ):
        """
        Args:
            csv_file: Đường dẫn đến processed_train.csv
            images_root: Folder chứa ảnh (data/images)
            telemetry_root: Folder chứa json (data/telemetry)
            tokenizer: Bộ mã hóa văn bản (BERT/GPT tokenizer...)
            transform: Các phép biến đổi ảnh (Resize, Normalize...)
            max_frames: Số lượng ảnh tối đa model sẽ xem (Input Size)
        """
        self.data = pd.read_csv(csv_file)
        self.img_root = images_root
        self.tel_root = telemetry_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_frames = max_frames
        self.future_steps = future_steps
        self.sample_fps = sample_fps
        self.source_fps = source_fps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row  = self.data.iloc[idx]
        video_id = row['video_id']
        start = float(row['start'])
        end = float(row['end'])
        
        # Đảm bảo caption là string, tránh lỗi nếu file CSV có ô trống
        caption = str(row['caption']) if pd.notna(row['caption']) else ""

        # --- 1. TÍNH ĐIỂM CẮT ---
        # Input: Start -> Mid
        # Target (Future): Mid -> End
        mid = start + (end - start) * 0.5
        
        # --- 2. INPUT FRAMES (Start -> Mid) ---
        # Paper-like sampling: n frames at 1 fps over [start, mid]
        sample_fps = int(self.sample_fps)
        source_fps = int(self.source_fps)

        if mid <= start:
            frame_times = np.array([start] * self.max_frames, dtype=float)
        else:
            step = 1.0 / float(sample_fps)
            candidates = np.arange(start, mid + 1e-9, step, dtype=float)
            if len(candidates) >= self.max_frames:
                pick_idx = np.linspace(0, len(candidates) - 1, num=self.max_frames)
                pick_idx = np.round(pick_idx).astype(int)
                frame_times = candidates[pick_idx]
            else:
                frame_times = np.linspace(start, mid, num=self.max_frames, dtype=float)

        frame_indices = frame_times
        video_tensors = []
        
        for t in frame_indices:
            # Stored frames are 5 fps exports: frame_1 = 0s, frame_2 = 0.2s...
            f_idx = int(t * float(source_fps)) + 1
            img_path = os.path.join(self.img_root, video_id, f"frame_{f_idx}.jpg")
            
            # Fallback nếu không có ảnh (màn hình đen)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                except:
                    img = Image.new('RGB', (160, 90), (0, 0, 0))
            else:
                img = Image.new('RGB', (160, 90), (0, 0, 0))
                
            if self.transform:
                # Resize ảnh về cùng kích thước (ví dụ: 160x90).
                img = self.transform(img) 
            video_tensors.append(img)
            
        video_input = torch.stack(video_tensors) # [16, 3, 160, 90]

        # --- 3. INPUT SENSOR & FUTURE MOTION ---
        json_path = os.path.join(self.tel_root, f"{video_id}.json")
        
        # Khởi tạo sensor data (3 chiều: speed, accel, course)
        sensor_data = []
        # CHỐT LẠI: Tương lai chỉ đoán 2 chiều (speed, course)
        future_motion = torch.zeros(self.future_steps, 2) 

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                full_log = data.get('locations', [])
                
                # Biến chốt chặn an toàn để không bị lỗi index out of range
                max_log_idx = max(0, len(full_log) - 1)
                
                # -- Lấy dữ liệu QUÁ KHỨ cho từng frame (Dùng 3 thông số) --
                for t in frame_indices:
                    # Telemetry is typically aligned per-second; round to nearest second.
                    idx = int(round(float(t)))
                    safe_idx = min(idx, max_log_idx)
                    
                    if len(full_log) > 0:
                        raw_speed = full_log[safe_idx].get('speed', 0)
                        raw_course = full_log[safe_idx].get('course', 0)
                        
                        # TÍNH GIA TỐC (dựa trên chênh lệch 1 giây)
                        if safe_idx > 0:
                            prev_speed = full_log[safe_idx - 1].get('speed', 0)
                            raw_accel = (raw_speed - prev_speed) / 1.0
                        else:
                            raw_accel = 0.0
                            
                        # Normalize
                        norm_speed = raw_speed / 30.0
                        norm_accel = raw_accel / 10.0
                        norm_course = raw_course / 360.0
                        
                        # Sensor quá khứ dùng cả 3
                        sensor_data.append([norm_speed, norm_accel, norm_course])
                    else:
                        sensor_data.append([0.0, 0.0, 0.0])

                # -- Lấy dữ liệu TƯƠNG LAI (Chỉ dùng 2 thông số) --
                future_indices = np.linspace(mid, end, num=self.future_steps)
                
                for i, t_val in enumerate(future_indices):
                    target_idx = int(round(float(t_val)))
                    safe_idx = min(target_idx, max_log_idx)
                    
                    if len(full_log) > 0:
                        raw_speed = full_log[safe_idx].get('speed', 0)
                        raw_course = full_log[safe_idx].get('course', 0)
                            
                        norm_speed = raw_speed / 30.0
                        norm_course = raw_course / 360.0
                        
                        # Tương lai chỉ lưu 2 thông số
                        future_motion[i] = torch.tensor([norm_speed, norm_course])
                        
            except Exception as e:
                # Nếu file JSON hỏng/rỗng, fallback về 0
                sensor_data = [[0.0, 0.0, 0.0] for _ in range(self.max_frames)]
        else:
            # Nếu không có file JSON, tạo sensor data mặc định
            sensor_data = [[0.0, 0.0, 0.0] for _ in range(self.max_frames)]
        
        # Convert sensor data thành tensor [max_frames, 3]
        sensor_input = torch.tensor(sensor_data, dtype=torch.float32)

        # --- 4. CAPTION (TARGET) ---
        tokens = self.tokenizer(
            caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt"
        )
        caption_ids = tokens["input_ids"].squeeze(0)

        return {
            'video': video_input,           # [16, 3, 160, 90]
            'sensor': sensor_input,         # [16, 3] (Speed, Accel, Course)
            'future_motion': future_motion, # [5, 2]  (Speed, Course) -> ĐÚNG CHUẨN
            'caption': caption_ids          # [30]
        }


        