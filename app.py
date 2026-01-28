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
