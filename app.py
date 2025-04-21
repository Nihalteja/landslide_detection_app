import streamlit as st
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load model once
@st.cache_resource
def load_model():
    model_path = "unet_model_v1.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# Normalize image for visualization
def normalize_img(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)
    img *= 255.0
    return img.astype(np.uint8)

# Prediction function
def predict_mask(img):
    resized = tf.image.resize(img, (128, 128))
    normalized = resized / 255.0
    prediction = model.predict(np.expand_dims(normalized, axis=0))[0]
    mask = (prediction > 0.5).astype(np.uint8).squeeze()
    return mask

# App UI
st.title("🌍 Landslide Detection from .h5 Files")
st.write("Upload a .h5 image file to view 14 bands and get a landslide prediction mask.")

uploaded_file = st.file_uploader("Upload a .h5 file", type=["h5"])

if uploaded_file is not None:
    try:
        with h5py.File(uploaded_file, 'r') as f:
            keys = list(f.keys())
            st.write("**Available keys in .h5:**", keys)

            if 'image' in keys:
                img = f['image'][:]
            else:
                img = f[keys[0]][:]

        st.write("**Image shape:**", img.shape)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Display all 14 bands
        st.subheader("🛰️ 14 Input Bands")
        band_cols = st.columns(7)
        for i in range(min(14, img.shape[-1])):
            band = normalize_img(img[:, :, i])
            with band_cols[i % 7]:
                st.image(band, caption=f"Band {i+1}", use_column_width=True, channels="L")

        # Show RGB Composite
        st.subheader("🌈 RGB Composite")
        if img.shape[-1] >= 3:
            rgb = normalize_img(img[:, :, :3])
        else:
            rgb = np.repeat(normalize_img(img[:, :, 0:1]), 3, axis=-1)
        st.image(rgb, caption="RGB Composite", use_column_width=True)

        # Pad to 14 channels if needed
        if img.shape[-1] < 14:
            img = np.pad(img, ((0, 0), (0, 0), (0, 14 - img.shape[-1])), mode='constant')
        elif img.shape[-1] > 14:
            img = img[:, :, :14]

        # Predict and show mask
        st.subheader("🧠 Predicted Landslide Mask")
        mask = predict_mask(img)
        st.image(mask * 255, caption="Predicted Mask", use_column_width=True, channels="L")

    except Exception as e:
        st.error(f"Error reading .h5 file: {e}")
