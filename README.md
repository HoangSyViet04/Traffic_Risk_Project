# Near-Future Driving Risk Captioning

> Hệ thống AI đa phương thức cho bài toán dự đoán rủi ro giao thông cận tương lai và sinh cảnh báo bằng ngôn ngữ tự nhiên.

## Abstract
Dự án xây dựng một mô hình Multi-Task Learning để giải đồng thời hai bài toán:
1. Dự đoán hành động tương lai của xe trong 5 bước (Speed, Course).
2. Sinh caption mô tả/cảnh báo tình huống giao thông.

Điểm quan trọng của đề tài là kết hợp thông tin hình ảnh từ camera hành trình với dữ liệu cảm biến chuyển động, từ đó vừa dự báo được xu hướng động lực học, vừa diễn giải bằng câu văn bản có ý nghĩa thực tế cho ADAS.

## Architecture
Mô hình gồm 3 khối chính:

### 1) Multimodal Encoder (Early Fusion)
1. Trích xuất đặc trưng ảnh bằng CNN 5 lớp Conv (bỏ lớp phân loại cuối).
2. Ghép đặc trưng ảnh với sensor (Speed, Accel, Course) theo từng timestep.
3. Đưa qua LSTM 2 tầng để lấy Context Vector 1024 chiều.

### 2) Action Regressor (LSTM + 3FC)
1. Nhận Context Vector.
2. Dự đoán trực tiếp 5 bước tương lai của xe (Speed, Course).
3. Đầu ra dạng `future_flat` rồi reshape thành `[Batch, 5, 2]`.

### 3) Near-Future Aware Caption Decoder
1. Ghép Context Vector với dự đoán tương lai.
2. Decoder sinh caption theo cơ chế auto-regressive.
	- Mặc định: Transformer decoder (masked self-attention + cross-attention vào context+future).
	- Tuỳ chọn: LSTM decoder (phiên bản cũ).
3. Caption mang tính “near-future aware”, không chỉ mô tả hiện tại mà còn phản ánh xu hướng sắp xảy ra.

### Training improvements (so với bản paper-cơ-bản)
- Chống mất cân bằng caption: token-weighted loss + weighted sampler.
- Ổn định train: AdamW/weight decay, label smoothing, LR scheduler, gradient clipping, AMP.
- Giảm overfit: dropout trong encoder/decoder/action head, early stopping mềm hơn (patience lớn + min_delta, có thể tắt).
- Đồng bộ thời gian telemetry: tự suy luận telemetry rate (1Hz/5fps/30fps) để tránh lệch nhịp.

## Results
Kết quả phụ thuộc cấu hình và dữ liệu (tokenizer/decoder/beam search). Khuyến nghị dùng beam search khi evaluate để tăng BLEU/CIDEr.

## Installation

### Cấu trúc thư mục và vị trí đặt dữ liệu
Người mới clone repo nên đảm bảo dữ liệu được đặt đúng chỗ như bên dưới để code chạy ngay:

```text
Traffic_Risk_Project/
├── train.py
├── evaluate.py
├── predict.py
├── plot_metrics.py
├── requirements.txt
├── data/
│   ├── processed_train.csv          # File train metadata
│   ├── test_data.csv                # File test metadata
│   ├── images/
│   │   └── <video_id>/
│   │       ├── frame_0.jpg
│   │       ├── frame_1.jpg
│   │       └── ...
│   └── telemetry/
│       ├── <video_id>.json
│       └── ...
├── saved_models/
│   ├── best_model.pth              # sinh ra sau khi train
│   └── training_log.csv
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── simple_tokenizer.py
│   └── models/
│       ├── encoder.py
│       ├── action_head.py
│       ├── decoder.py
│       ├── transformer_decoder.py
│       └── full_model.py
```

Lưu ý nhanh:
1. Ảnh đã resize đặt trong `data/images/<video_id>/`.
2. Telemetry JSON đặt trong `data/telemetry/`.
3. CSV train/test đặt đúng trong `data/`.

### link dataset: 
```bash
https://drive.google.com/file/d/1aICWnxlVep27B_9pudZoJbAUPKVVZmmN/view
```

### 1) Clone repository
```bash
git clone https://github.com/HoangSyViet04/Traffic_Risk_Project.git
```

### 2) Tạo môi trường và cài dependencies
```bash
python -m venv .venv
```

Nếu dùng CMD:
```bash
.venv\Scripts\activate
```

Nếu dùng Linux/macOS:
```bash
source .venv/bin/activate
```

Hoặc cài theo file sẵn có:
```bash
pip install -r requirements.txt
```

## Usage

### 1) Huấn luyện mô hình
```bash
python train.py
```
Chạy toàn bộ vòng lặp train/val, lưu model tốt nhất vào `saved_models/best_model.pth`.

Gợi ý cấu hình trong `src/config.py`:
- `DECODER_TYPE = "transformer"` (khuyến nghị)
- `TOKENIZER_TYPE = "simple"` (thường giúp caption metrics tốt hơn với dataset nhỏ)
- Nếu bị dừng sớm: tăng `EARLY_STOPPING_PATIENCE` hoặc đặt `USE_EARLY_STOPPING = False`

### 2) Đánh giá trên toàn tập test
```bash
python evaluate.py --beam-size 3 --length-penalty 0.7
```
Tính đầy đủ các chỉ số: MSE, BLEU-4, METEOR, CIDEr.

### 3) Dự đoán 1 mẫu đơn lẻ
```bash
python predict.py --index 0 --beam-size 3 --length-penalty 0.7
```
In ra console:
1. Dự đoán hành động tương lai (Speed, Course).
2. Câu cảnh báo/caption do mô hình sinh ra.

### 4) Vẽ Learning Curve
```bash
python plot_metrics.py
```
Sinh biểu đồ học tập từ `training_log.csv` để phân tích quá trình hội tụ.

## Acknowledgments
Ý tưởng và định hướng học thuật của dự án được tham chiếu từ bài báo:

**"Image Captioning in Near Future from Vehicle Camera Images and Motion Information"**
