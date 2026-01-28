import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = "/content/drive/MyDrive/skin_type_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['dry', 'normal', 'oily']

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Skin Type Recognition", layout="centered")

st.title("Skin Type Recognition Dashboard")
st.write("Upload a facial image to predict skin type.")

# ---------------------------
# Image uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader("Prediction Result")
    st.write(f"**Skin Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_type_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

