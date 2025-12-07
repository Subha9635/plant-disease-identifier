import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------------
# MODEL LOADING AND SETUP
# ----------------------------------------------------------------------------------

# 1. Load Model (Cached to prevent reloading)
@st.cache_resource
def load_model():
    # Make sure this file matches the one you downloaded from Colab
    return tf.keras.models.load_model('plant_disease_model.h5')

model = load_model()

# 2. Define Classes (Updated to match YOUR specific Colab training data)
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

CONFIDENCE_THRESHOLD = 60.0 # Adjusted for the new model

# ----------------------------------------------------------------------------------
# UI AND INPUT LOGIC
# ----------------------------------------------------------------------------------

st.set_page_config(page_title="Plant Disease Scanner", layout="wide")
st.title("🌿 Plant Disease Identifier")
st.markdown("Scan a leaf using your camera or upload an image.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📁 1. Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    camera_input = st.camera_input("📸 2. Scan Leaf")

# Determine input source
if uploaded_file is not None:
    source_file = uploaded_file
elif camera_input is not None:
    source_file = camera_input
else:
    source_file = None

if source_file is not None:
    
    # --- Image Processing ---
    image = Image.open(source_file)
    st.image(image, caption="Input Image", width=300)
    
    # Resize to 224x224
    # Using ImageOps.fit ensures we don't distort the aspect ratio
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # ⚠️ CRITICAL CHANGE FOR YOUR MODEL:
    # We do NOT divide by 255.0 here because your model has a Rescaling layer inside it.
    img_array = np.expand_dims(img_array, axis=0) 

    # --- Prediction ---
    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    st.markdown("---")
    
    # --- Results ---
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(f"⚠️ Low Confidence ({confidence:.2f}%). Are you sure this is a plant leaf?")
    
    elif predicted_class == 'Background_without_leaves':
        st.error("❌ No leaf detected. Please verify the image.")
        
    else:
        clean_name = predicted_class.replace("___", " - ").replace("_", " ")
        st.subheader(f"Result: **{clean_name}**")
        st.caption(f"Confidence: **{confidence:.2f}%**")
        
        if "healthy" in predicted_class.lower():
            st.balloons()
            st.success("✅ Plant appears healthy.")
        else:
            st.error("🚨 Disease Detected.")
            st.info("Check recommended treatments for this condition.")
