import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def EEGNet_SSVEP(input_shape, num_classes, F1=32, D=2, F2=64, dropout_rate=0.5, kernLength=64, reg=1e-4):
    """Mô hình EEGNet tối ưu cho tín hiệu BCI"""
    time_steps, ch = input_shape
    inputs = layers.Input(shape=(time_steps, ch))
    x = layers.Reshape((time_steps, ch, 1))(inputs)
    
    x = layers.Conv2D(F1, (kernLength, 1), padding='same', kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.DepthwiseConv2D((1, ch), depth_multiplier=D, depthwise_regularizer=regularizers.l2(reg), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((4, 1))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.SeparableConv2D(F2, (16, 1), padding='same', depthwise_regularizer=regularizers.l2(reg), pointwise_regularizer=regularizers.l2(reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((8, 1))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.Activation('elu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return models.Model(inputs=inputs, outputs=outputs)


def ShallowConvNet(input_shape, num_classes, dropout_rate=0.5):
    """Mô hình ShallowConvNet trích xuất năng lượng dải sóng (Bandpower)"""
    time_steps, ch = input_shape
    inputs = layers.Input(shape=(time_steps, ch))
    x = layers.Reshape((time_steps, ch, 1))(inputs)
    
    # 1. Temporal Convolution
    x = layers.Conv2D(40, (25, 1), kernel_regularizer=regularizers.l2(1e-4))(x)
    
    # 2. Spatial Convolution
    x = layers.Conv2D(40, (1, ch), use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    
    # 3. Squaring (Tính bình phương năng lượng)
    x = layers.Lambda(lambda z: tf.square(z))(x)
    
    # 4. Average Pooling
    x = layers.AveragePooling2D((75, 1), strides=(15, 1))(x)
    
    # 5. Logarithmic Activation
    x = layers.Lambda(lambda z: tf.math.log(tf.clip_by_value(tf.cast(z, tf.float32), 1e-5, 10000.0)))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 6. Phân loại
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)