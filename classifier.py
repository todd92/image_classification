# Quick example to grab training data automatically
# from bing_image_downloader import downloader

# downloader.download("golden retriever dog", limit=50, output_dir="train_data")
# downloader.download("mountain landscape", limit=50, output_dir="train_data")
# downloader.download("sports car", limit=50, output_dir="train_data")


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. SETTINGS
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "train_data"

# 2. LOAD DATA
# This automatically labels the images based on the folder names
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Detected classes: {class_names}")

# 3. DATA AUGMENTATION (Helps prevent overfitting)
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

# 4. BUILD THE MODEL (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Freeze the pre-trained layers

model = models.Sequential(
    [
        data_augmentation,
        layers.Rescaling(1.0 / 127.5, offset=-1),  # Normalize pixels for MobileNet
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(class_names), activation="softmax"),
    ]
)

# 5. COMPILE AND TRAIN
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Starting training...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6. SAVE THE MODEL
model.save("my_classifier.h5")
print("Model saved as my_classifier.h5")
