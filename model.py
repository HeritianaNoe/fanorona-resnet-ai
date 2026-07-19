import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

ROWS = 5
COLS = 9
BOARD_SHAPE = (ROWS, COLS, 1)
NUM_DIRECTIONS = 8
MAX_ACTIONS = ROWS * COLS * NUM_DIRECTIONS

def residual_block(x, filters, kernel_size=3):
    res = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    res = layers.BatchNormalization()(res)
    res = layers.Activation('relu')(res)
    res = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(1e-4))(res)
    res = layers.BatchNormalization()(res)
    out = layers.add([x, res])
    out = layers.Activation('relu')(out)
    return out

def create_massive_fanorona_model(num_residual_blocks=10):
    inputs = layers.Input(shape=BOARD_SHAPE, name="board_input")
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    for _ in range(num_residual_blocks):
        x = residual_block(x, filters=256)
        
    policy_x = layers.Conv2D(2, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    policy_x = layers.BatchNormalization()(policy_x)
    policy_x = layers.Activation('relu')(policy_x)
    policy_x = layers.Flatten()(policy_x)
    policy_output = layers.Dense(MAX_ACTIONS, activation='softmax', name='policy_output')(policy_x)
    
    value_x = layers.Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    value_x = layers.BatchNormalization()(value_x)
    value_x = layers.Activation('relu')(value_x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(value_x)
    value_output = layers.Dense(1, activation='tanh', name='value_output')(value_x)
    
    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
        loss_weights={'policy_output': 1.0, 'value_output': 1.0}
    )
    return model

if __name__ == "__main__":
    fanorona_brain = create_massive_fanorona_model(num_residual_blocks=10)
    fanorona_brain.summary()
    
    fanorona_brain.save("fanorona_resnet_model.h5")
    print("[OK] Model saved as fanorona_resnet_model.h5")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(fanorona_brain)
    tflite_model = converter.convert()
    with open("fanorona_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("[OK] Model converted and saved as fanorona_model.tflite")
