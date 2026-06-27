import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Nạp cấu hình từ trung tâm điều khiển
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *

def fine_tune_outlier(subj='S003'):
    print(f"🚀 BẮT ĐẦU FINE-TUNING CÁ NHÂN HÓA CHO SUBJECT {subj}...")
    
    # 1. Định vị mô hình gốc đã train từ train.py
    model_path = os.path.join(MODEL_SAVE_DIR, f"shallow_best_{subj}.h5")
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model gốc tại {model_path}. Hãy chạy train.py trước!")
        return

    # 2. Nạp thông tin cấu trúc dữ liệu từ Memmap
    y_read = np.load(MEMMAP_Y)
    subject_read = np.load(SUBJECT_MEMMAP)
    num_classes = len(np.unique(y_read))
    epoch_len_samples = int(EPOCH_LEN_SEC * SFREQ_TARGET)
    n_channels = 64
    
    X_read = np.memmap(MEMMAP_X, mode="r", dtype=DTYPE, shape=(len(y_read), epoch_len_samples, n_channels))

    # 3. Trích xuất và chia dữ liệu của riêng Subj này (20% Học thêm, 80% Test)
    subj_idx = np.where(subject_read == subj)[0]
    calib_idx, test_idx = train_test_split(
        subj_idx, 
        test_size=0.8, 
        random_state=RANDOM_SEED, 
        stratify=y_read[subj_idx]
    )

    print(f"- Số lượng mẫu dùng để máy làm quen (20%): {len(calib_idx)} epochs")
    print(f"- Số lượng mẫu để bài test thực tế (80%): {len(test_idx)} epochs")

    # 4. TẠO DATASET IN-MEMORY (Vì dữ liệu rất nhỏ, nạp thẳng vào RAM để đạt tốc độ tối đa)
    def create_memory_dataset(indices, augment=False):
        # Trích xuất thẳng thành Numpy Array
        X_data = X_read[indices].astype(np.float32)
        y_data = tf.one_hot(y_read[indices], depth=num_classes)
        
        ds = tf.data.Dataset.from_tensor_slices((X_data, y_data))
        if augment:
            def augment_fn(x, y):
                x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)
                return x, y
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(1024).repeat()
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    calib_ds = create_memory_dataset(calib_idx, augment=True)
    test_ds = create_memory_dataset(test_idx, augment=False)

    # 5. Kiểm tra sức mạnh của mô hình TRƯỚC khi Fine-tune
    print("\nĐang kiểm tra độ chính xác Baseline (LOSO-CV Gốc)...")
    model = tf.keras.models.load_model(model_path)
    y_pred_before = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    acc_before = np.mean(y_pred_before == y_read[test_idx])
    
    # 6. Cấu hình Fine-tuning (Learning Rate 1e-4 để chống Catastrophic Forgetting)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    ]
    
    print("\n🧠 Đang tiến hành Hiệu chuẩn (Fine-Tuning)...")
    model.fit(
        calib_ds, 
        validation_data=test_ds, 
        steps_per_epoch=max(1, len(calib_idx) // BATCH_SIZE),
        epochs=40, 
        callbacks=callbacks, 
        verbose=2
    )

    # 7. Kiểm tra lại SAU khi Fine-tune
    y_pred_after = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    acc_after = np.mean(y_pred_after == y_read[test_idx])

    # 8. Báo cáo kết quả
    print("\n" + "="*50)
    print(f"📉 Accuracy TRƯỚC Hiệu chuẩn: {acc_before:.4f} ({(acc_before*100):.2f}%)")
    print(f"📈 Accuracy SAU Hiệu chuẩn:   {acc_after:.4f} ({(acc_after*100):.2f}%)")
    print(f"🔥 MỨC TĂNG TRƯỞNG:           +{(acc_after - acc_before)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    fine_tune_outlier('S003')