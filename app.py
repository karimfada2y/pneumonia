import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Download model from Google Drive using gdown
model_path = "pneumonia_model.h5"
if not os.path.exists(model_path):
    import gdown
    url = "https://drive.google.com/uc?id=1LH2oI1h9SXtt40vNyjGrf_21w8nFuUA-"
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Streamlit UI
st.title("🩻 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image, and the model will predict if it is **Normal** or shows **Pneumonia**.")

uploaded_file = st.file_uploader("Upload a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.success("✅ Prediction: **Normal**")
    else:
        st.error("⚠️ Prediction: **Pneumonia**")
