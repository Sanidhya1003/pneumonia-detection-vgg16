import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

MODEL_PATH = "models/pneumonia_vgg16.h5"

def predict_image(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return

    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return

    model = load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_data = preprocess_input(img_array)

    prediction = model.predict(img_data)

    if prediction[0][0] > prediction[0][1]:
        print("Prediction: Person is safe.")
    else:
        print("Prediction: Person is affected with Pneumonia.")

    print(f"Raw prediction scores: {prediction}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <path_to_image>")
    else:
        predict_image(sys.argv[1])