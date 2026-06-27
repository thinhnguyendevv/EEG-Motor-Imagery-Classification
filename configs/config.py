import os
import numpy as np

# ==============================================================================
# HỆ THỐNG ĐƯỜNG DẪN (PATHS)
# Tự động lấy thư mục gốc của project làm chuẩn để chạy ở máy nào cũng không lỗi
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dữ liệu gốc và Dữ liệu đã xử lý
DATA_DIR = os.path.join(BASE_DIR, "data", "files")
WORKING_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_DIR = os.path.join(WORKING_DIR, "saved_models")

# Tự động tạo thư mục nếu chưa có
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Các file lưu trữ Memmap
MEMMAP_X = os.path.join(WORKING_DIR, "X_eeg_memmap.dat")
MEMMAP_Y = os.path.join(WORKING_DIR, "y_eeg.npy")
SUBJECT_MEMMAP = os.path.join(WORKING_DIR, "subject_per_epoch.npy")

# ==============================================================================
# THÔNG SỐ TIỀN XỬ LÝ TÍN HIỆU EEG (SIGNAL PROCESSING)
# ==============================================================================
SFREQ_TARGET = 160            # Tần số lấy mẫu mục tiêu (Hz)
EPOCH_LEN_SEC = 2.0           # Độ dài mỗi cửa sổ thời gian (giây)
BANDPASS = (8.0, 30.0)        # Lọc băng thông: Lấy dải sóng Mu và Beta (Motor Imagery)
NOTCH_FREQ = 50.0             # Lọc nhiễu điện lưới (50Hz)

# ==============================================================================
# THÔNG SỐ HUẤN LUYỆN MÔ HÌNH (TRAINING HYPERPARAMETERS)
# ==============================================================================
BATCH_SIZE = 64
EPOCHS = 80
DTYPE = np.float32            # Cấu hình kiểu dữ liệu để tối ưu RAM
RANDOM_SEED = 42

# Cố định Seed để kết quả chạy lại luôn giống nhau (Reproducibility)
np.random.seed(RANDOM_SEED)