import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# 1. ADD THIS IMPORT
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("my_classifier.h5")
class_names = ["cars", "dogs", "landscapes"]


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 2. REPLACE YOUR OLD MATH WITH THIS LINE
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)

    # 3. USE ARGMAX DIRECTLY ON THE PREDICTION
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    print(f"Image: {os.path.basename(img_path)}")
    print(
        f"Prediction: {class_names[predicted_index]} ({100 * confidence:.2f}% confidence)\n"
    )


# ... rest of your loop remains the same

# 3. Test it on your personal pictures
# Point this to wherever you stored your 8-10 personal photos
my_photos_path = "./personal_photos/"

print("--- Running Predictions ---")
for img_file in os.listdir(my_photos_path):
    if img_file.endswith((".jpg", ".png", ".jpeg")):
        predict_image(os.path.join(my_photos_path, img_file))
