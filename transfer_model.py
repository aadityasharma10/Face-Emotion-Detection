import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

train_dir = "train"
test_dir = "test"

# Load dataset and convert grayscale to RGB automatically
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(160,160),
    batch_size=32,
    color_mode="rgb"
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(160,160),
    batch_size=32,
    color_mode="rgb"
)

# Preprocess for MobileNet
train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
val_data = val_data.map(lambda x, y: (preprocess_input(x), y))

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(160,160,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze base model

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("mobilenet_model.h5")

print("Transfer Learning Training Complete")

# Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('MobileNet Accuracy')
plt.legend(['Train','Validation'])
plt.show()