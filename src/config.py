import torch
import os

basedir  = os.path.dirname(os.path.dirname(__file__))
class Config:
    # Đường dẫn 
    TRAIN_CSV = os.path.join(basedir, 'data', 'processed_train.csv')
    TEST_CSV = os.path.join(basedir, 'data', 'test_data.csv')
    IMAGES_ROOT = os.path.join(basedir, 'data', 'images')
    TELEMETRY_ROOT = os.path.join(basedir, 'data', 'telemetry')
    VOCAB_PATH = os.path.join(basedir, "saved_models", "custom_vocab.json")

    # cấu hình model
    # Kích thước ảnh đầu vào 
    IMAGE_SIZE = (90, 160)

    # Số frame model sẽ nhìn (Start -> Mid)
    # Paper setting: n = 5 frames at 1 fps (observed from past to present)
    MAX_FRAMES = 5

    # Frame naming in this repo follows 5 fps exports: frame_{int(t*5)+1}.jpg
    SOURCE_FPS = 5
    # Sampling rate for selecting input frames
    SAMPLE_FPS = 1

    # Kích thước vector nhúng (Embedding cho word)
    EMBED_SIZE = 256
    # Kích thước hidden state cho LSTM (Encoder + Decoder)
    HIDDEN_SIZE = 1024

    # Số lượng tham số sensor (Speed, Acceleration, Course)
    SENSOR_DIM = 3

    # Action Regressor dự đoán số bước tương lai
    FUTURE_STEPS = 5

    # --- HUẤN LUYỆN ---
    BATCH_SIZE = 50
    # Paper uses 30 epochs
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3

    # Multi-task loss weights
    CAPTION_LOSS_WEIGHT = 1.0
    MOTION_LOSS_WEIGHT = 1.0

    # Caption loss regularization
    LABEL_SMOOTHING = 0.0  # set to 0.05-0.1 to potentially improve generalization

    # LR scheduler (ReduceLROnPlateau)
    USE_LR_SCHEDULER = True
    LR_SCHED_FACTOR = 0.5
    LR_SCHED_PATIENCE = 2
    MIN_LR = 1e-6

    # Encoder CNN behavior
    FREEZE_CNN = True

    # Decoding (affects BLEU-4/CIDEr at inference)
    BEAM_SIZE = 5
    LENGTH_PENALTY_ALPHA = 0.7
    MIN_DECODE_LEN = 3

    # Thiết bị (tự động chọn GPU nếu có)
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SAVE 
    MODEL_SAVE_PATH = 'saved_models/best_model.pth'