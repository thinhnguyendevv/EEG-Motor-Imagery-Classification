import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# Nạp các module từ project của bạn
from models import ShallowConvNet
from data_preprocessing import (
    prepare_data, get_tf_dataset, 
    WORKING_DIR, MEMMAP_X, MEMMAP_Y, SUBJECT_MEMMAP, 
    EPOCH_LEN_SEC, SFREQ_TARGET, BATCH_SIZE, DTYPE
)

# Mixed Precision cho GPU
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

MODEL_SAVE_DIR = os.path.join(WORKING_DIR, "saved_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
EPOCHS = 80

def plot_training(history, subj):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'Accuracy - {subj}')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'Loss - {subj}')
    plt.legend()
    plt.show()

def run_loso_cv():
    prepare_data() # Đảm bảo data đã sẵn sàng
    
    y_read = np.load(MEMMAP_Y)
    subject_read = np.load(SUBJECT_MEMMAP)
    unique_subjects = np.unique(subject_read)
    
    epoch_len_samples = int(EPOCH_LEN_SEC * SFREQ_TARGET)
    n_channels = 64
    num_classes = len(np.unique(y_read))
    
    X_read = np.memmap(MEMMAP_X, mode="r", dtype=DTYPE, shape=(len(y_read), epoch_len_samples, n_channels))
    all_acc = []

    # Chạy thử 3 Subject đầu tiên
    for subj in unique_subjects[:3]: 
        print(f"\n{'='*20} LOSO Fold: Testing on Subject {subj} {'='*20}")
        
        model_path = os.path.join(MODEL_SAVE_DIR, f"shallow_best_{subj}.h5")
        # Lấy danh sách tên người cho tập Train và tập Test
        train_subjects = [s for s in unique_subjects if s != subj]
        test_subjects = [subj]

        # Truyền trực tiếp danh sách tên vào hàm để TFRecord xử lý
        train_ds = get_tf_dataset(train_subjects, num_classes, epoch_len_samples, n_channels, augment=True)
        test_ds = get_tf_dataset(test_subjects, num_classes, epoch_len_samples, n_channels, augment=False)
        if not os.path.exists(model_path):
            class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_read[train_idx]), y=y_read[train_idx])))
            model = ShallowConvNet((epoch_len_samples, n_channels), num_classes)
            
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=EPOCHS * (len(train_idx) // BATCH_SIZE))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0), 
                          loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            ]

            history = model.fit(train_ds, validation_data=test_ds, steps_per_epoch=max(1, len(train_idx) // BATCH_SIZE),
                                epochs=EPOCHS, class_weight=class_weights_dict, callbacks=callbacks, verbose=2)
            plot_training(history, subj)
            loaded_model = model
        else:
            print(f"✅ Đã có model cho {subj}, đang đánh giá...")
            loaded_model = tf.keras.models.load_model(model_path)

        y_pred = np.argmax(loaded_model.predict(test_ds), axis=1)
        acc = np.mean(y_pred == y_read[test_idx])
        all_acc.append(acc)
        print(f"Test Accuracy for subject {subj}: {acc:.4f}")

    if len(all_acc) > 0:
        print(f"\n🚀 AVERAGE LOSO-CV ACCURACY: {np.mean(all_acc):.4f}")

if __name__ == "__main__":
    run_loso_cv()
   