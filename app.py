import streamlit as st
import tensorflow as tf
import os
import gdown

st.title("Skin Type Recognition")

MODEL_PATH = "skin_type_model.h5"

# Show current files (debug)
st.write("Files before download:", os.listdir())

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(
        "https://drive.google.com/drive/u/0/folders/1FHZI6lBFGfUt7xb4aXY4q8rBV2PzbpXU",
        MODEL_PATH,
        quiet=False
    )

st.write("Files after download:", os.listdir())

# NOW load the model (this must be AFTER download)
model = tf.keras.models.load_model(MODEL_PATH)

st.success("Model loaded successfully 🎉")
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.title("Skin Type Recognition")
st.success("App started successfully ✅")

MODEL_PATH = "skin_type_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.info("Model loaded 🎉")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_names = ["Dry", "Normal", "Oily"]

    st.success(f"Predicted Skin Type: **{class_names[np.argmax(prediction)]}**")

