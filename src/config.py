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
    MAX_FRAMES = 16

    # Kích thước vector nhúng (Embedding cho word)
    EMBED_SIZE = 256
    # Kích thước hidden state cho LSTM (Encoder + Decoder)
    HIDDEN_SIZE = 1024

    # --- ENCODER SEQUENCE BACKBONE ---
    # Options: "transformer" (default), "lstm", "bilstm", "gru"
    ENCODER_TYPE = "transformer"

    # Transformer encoder hyperparameters (only used when ENCODER_TYPE="transformer")
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_FF_DIM = 2048
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_USE_CLS_TOKEN = True

    # Số lượng tham số sensor (Speed, Acceleration, Course)
    SENSOR_DIM = 3

    # Action Regressor dự đoán số bước tương lai
    FUTURE_STEPS = 5

    # --- HUẤN LUYỆN ---
    BATCH_SIZE = 50
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3

    # Loss weights (tune to balance objectives)
    LOSS_CAPTION_WEIGHT = 1.0
    LOSS_MOTION_WEIGHT = 1.0

    # Thiết bị (tự động chọn GPU nếu có)
    DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SAVE 
    MODEL_SAVE_PATH = 'saved_models/best_model.pth'