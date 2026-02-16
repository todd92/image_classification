import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model and set categories
model = tf.keras.models.load_model("my_classifier.h5")
class_names = ["cars", "dogs", "landscapes"]

# Setup failure directory
failure_dir = "failure_gallery"
if not os.path.exists(failure_dir):
    os.makedirs(failure_dir)


def analyze_failures(test_path):
    for img_file in os.listdir(test_path):
        if not img_file.endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(test_path, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array, verbose=0)
        idx = np.argmax(preds[0])
        conf = preds[0][idx]

        print(
            f"Checking {img_file}: Predicted {class_names[idx]} with {conf:.2f} confidence"
        )

        # If confidence is low (< 60%), save it for review
        if conf < 0.80:
            print(f"Low confidence ({conf:.2f}) for {img_file}. Copying to gallery...")
            shutil.copy(img_path, os.path.join(failure_dir, f"low_conf_{img_file}"))


analyze_failures("./personal_photos/")
