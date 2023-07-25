import tensorflow as tf
import numpy as np

with open('Raw_images.npy', 'rb') as f:
    X = np.load(f, allow_pickle=True)
    y = np.load(f, allow_pickle=True)
print(X.shape, y.shape)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, activation='relu', padding='same',
                              input_shape=(256,256, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

model.compile(optimizer=tf.keras.optimizers.experimental.SGD(momentum=0.9),
                loss= tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1))

model.fit(x=X, y=y, validation_split=0.2,
    epochs=3,
    batch_size=32)

# First try it works:)