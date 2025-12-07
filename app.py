import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- SET UP ---
# Hide warnings to keep the app clean
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONSTANTS ---
MODEL_PATH = 'plant_disease_model.h5'
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --- UI ---
st.title("🌿 Plant Disease Identifier")
st.markdown("Upload a plant leaf image to detect diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Analyze'):
        with st.spinner('Processing...'):
            try:
                model = load_model()

                # Preprocess
                img = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array)
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                confidence = 100 * np.max(predictions[0])

                # Output
                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

            except Exception as e:
                st.error(f"Error: {e}")
