from tensorflow.keras.preprocessing.image import ImageDataGenerator


import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load the pre-trained brain (Base)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Freeze it so it doesn't forget what it knows

# 2. Build the "Head" and combine them into 'model'
model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(3, activation="softmax"),  # 3 for your 3 categories
    ]
)

# 3. Compile (The 'instruction manual' for training)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# NOW you can use it!
model.summary()


# 1. We tell the generator to set aside 20% of images for the "Quiz"
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # <--- This is the "Vault"
)

# 2. This is the "Study Material" (80% of your photos)
train_generator = datagen.flow_from_directory(
    "./train_data/",
    target_size=(224, 224),
    batch_size=32,
    subset="training",  # <--- Use the 80%
)

# 3. This is the "Surprise Quiz" (20% of your photos)
validation_generator = datagen.flow_from_directory(
    "./train_data/",
    target_size=(224, 224),
    batch_size=32,
    subset="validation",  # <--- Use the 20% from the vault
)


print("Starting training...")  # Let's add a clear starting message

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    verbose=1,  # <--- This 1 tells TensorFlow to show the progress bar
)

print("Training finished! Saving model...")
model.save("my_classifier_v2.h5")
