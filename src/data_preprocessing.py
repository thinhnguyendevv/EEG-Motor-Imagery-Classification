import sys
import os
import numpy as np
import mne
import joblib
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Thêm đường dẫn thư mục gốc để nạp cấu hình
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *

# Thư mục chứa các file nhị phân TFRecord (Tự động tạo)
TFRECORD_DIR = os.path.join(WORKING_DIR, "tfrecords")
os.makedirs(TFRECORD_DIR, exist_ok=True)

def read_annotations_safe(raw):
    try:
        events, event_ids = mne.events_from_annotations(raw, verbose=False)
        return events, event_ids
    except Exception:
        return np.zeros((0, 3), dtype=int), {}

def prepare_data():
    """Hàm trích xuất dữ liệu thô sang Memmap (Chạy 1 lần duy nhất)"""
    if os.path.exists(MEMMAP_X) and os.path.exists(MEMMAP_Y):
        print("✅ Đã tìm thấy dữ liệu Memmap, bỏ qua bước trích xuất!")
    else:
        print("⏳ Đang xử lý dữ liệu thô sang Memmap...")
        edf_files = sorted([os.path.join(r, f) for r, _, files in os.walk(DATA_DIR) for f in files if f.lower().endswith(".edf")])
        if not edf_files: raise RuntimeError(f"Không tìm thấy file .edf trong {DATA_DIR}")
        
        n_channels, total_epochs = None, 0
        for f in tqdm(edf_files, desc="Scanning headers"):
            try:
                raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
                if n_channels is None: n_channels = len(raw.ch_names)
                events, _ = read_annotations_safe(raw)
                if events.shape[0] > 0: total_epochs += events.shape[0] * 3 
                raw.close()
            except: continue

        epoch_len_samples = int(EPOCH_LEN_SEC * SFREQ_TARGET)
        X_mm = np.memmap(MEMMAP_X, mode="w+", dtype=DTYPE, shape=(total_epochs, epoch_len_samples, n_channels))
        y_labels, subject_per_epoch, pos = [], [], 0
        offsets = [0, int(0.5 * SFREQ_TARGET), int(1.0 * SFREQ_TARGET)] 

        for f in tqdm(edf_files, desc="Extracting epochs"):
            subj_id = os.path.basename(os.path.dirname(f))
            try:
                raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                if int(raw.info['sfreq']) != SFREQ_TARGET: raw.resample(SFREQ_TARGET)
                events, event_ids = read_annotations_safe(raw)
                inv_event_ids = {v: k for k, v in event_ids.items()}
                
                for ev in events:
                    for offset in offsets:
                        start, stop = int(ev[0]) + offset, int(ev[0]) + offset + epoch_len_samples
                        if stop > raw.n_times: continue
                        arr = np.asarray(raw[:, start:stop][0], dtype=DTYPE).T
                        try:
                            arr_T = mne.filter.notch_filter(arr.T, SFREQ_TARGET, freqs=NOTCH_FREQ, verbose=False)
                            arr_T = mne.filter.filter_data(arr_T, SFREQ_TARGET, l_freq=BANDPASS[0], h_freq=BANDPASS[1], verbose=False)
                            arr = arr_T.T.astype(DTYPE)
                        except: pass
                        arr = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-8)
                        if pos < total_epochs:
                            X_mm[pos] = arr
                            y_labels.append(str(inv_event_ids.get(int(ev[2]), "UNKNOWN")))
                            subject_per_epoch.append(subj_id)
                            pos += 1
                raw.close()
            except: continue
        X_mm.flush()
        if pos < total_epochs:
            X_mm = np.memmap(MEMMAP_X, mode="r+", dtype=DTYPE, shape=(pos, epoch_len_samples, n_channels))
        
        le = LabelEncoder()
        np.save(MEMMAP_Y, le.fit_transform(np.array(y_labels)))
        np.save(SUBJECT_MEMMAP, np.array(subject_per_epoch))
        joblib.dump(le, os.path.join(WORKING_DIR, "label_encoder.joblib"))
        print("✅ Đã tạo xong Memmap Pipeline!")

    # LUÔN KIỂM TRA VÀ TẠO TFRECORD ĐỂ HUẤN LUYỆN SIÊU TỐC
    create_tfrecords_from_memmap()

def create_tfrecords_from_memmap():
    """Đóng gói dữ liệu của từng người thành các file nhị phân riêng biệt (Chạy 1 lần)"""
    y_read = np.load(MEMMAP_Y)
    subject_read = np.load(SUBJECT_MEMMAP)
    epoch_len_samples = int(EPOCH_LEN_SEC * SFREQ_TARGET)
    n_channels = 64
    
    unique_subjects = np.unique(subject_read)
    X_read = np.memmap(MEMMAP_X, mode="r", dtype=DTYPE, shape=(len(y_read), epoch_len_samples, n_channels))

    print("\n🚀 Chuẩn bị dữ liệu TFRecord (Native C++ Multithreading)...")
    for subj in tqdm(unique_subjects, desc="Writing TFRecords"):
        tfrecord_path = os.path.join(TFRECORD_DIR, f"{subj}.tfrecord")
        if os.path.exists(tfrecord_path):
            continue  # Nếu file đã tồn tại thì bỏ qua để tiết kiệm thời gian
            
        subj_idxs = np.where(subject_read == subj)[0]
        
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for idx in subj_idxs:
                eeg_bytes = X_read[idx].tobytes()
                label = int(y_read[idx])
                
                feature = {
                    'eeg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[eeg_bytes])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def get_tf_dataset(subjects_list, num_classes, epoch_len_samples, n_channels, augment=False):
    """Đọc dữ liệu đa luồng trực tiếp từ ổ cứng bằng mã nguồn C++ của TensorFlow"""
    # Lấy đường dẫn các file TFRecord tương ứng với danh sách Subject truyền vào
    files = [os.path.join(TFRECORD_DIR, f"{s}.tfrecord") for s in subjects_list]
    files = [f for f in files if os.path.exists(f)]
    
    # Kích hoạt chế độ đọc song song từ nhiều file cùng lúc
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    def parse_tfrecord_fn(example_proto):
        feature_description = {
            'eeg': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        # Giải mã chuỗi bytes thành mảng float32 và định hình lại
        eeg = tf.io.decode_raw(parsed['eeg'], out_type=tf.float32)
        eeg = tf.reshape(eeg, [epoch_len_samples, n_channels])
        label = tf.one_hot(parsed['label'], depth=num_classes)
        return eeg, label

    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def augment_fn(x, y):
            x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)
            return x, y
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(4096)
        ds = ds.repeat()

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)