import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------------
# MODEL LOADING AND SETUP (Same as before)
# ----------------------------------------------------------------------------------

# 1. Load Model (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_village_model.h5')

model = load_model()

# 2. Define Classes (Using your provided list)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# Set a threshold to filter out uncertain images (like human hands/non-plant objects)
CONFIDENCE_THRESHOLD = 0.75 # 75% confidence needed to make a positive ID

# ----------------------------------------------------------------------------------
# UI AND INPUT LOGIC (FIXED)
# ----------------------------------------------------------------------------------

st.set_page_config(page_title="Plant Disease Scanner", layout="wide")
st.title("🌿 PlantVillage AI Disease Scanner")
st.markdown("Scan a leaf using your camera or upload an image from your gallery.")

st.markdown("---")

# Use columns to present both input options clearly
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📁 1. Upload Image from Gallery", type=["jpg", "jpeg", "png"])

with col2:
    camera_input = st.camera_input("📸 2. Scan Leaf with Camera")

# Determine which input source to use
if uploaded_file is not None:
    source_file = uploaded_file
elif camera_input is not None:
    source_file = camera_input
else:
    source_file = None

if source_file is not None:
    
    # --- Image Processing ---
    image = Image.open(source_file)
    st.image(image, caption="Captured/Uploaded Image", width=300)
    
    # Resize and Normalize (Must match training size: 224x224)
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 

    # --- Prediction ---
    with st.spinner('Analyzing image...'):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    st.markdown("---")
    
    # --- Misclassification Fix: Apply Confidence Threshold ---
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Non-Plant Object Detected or Highly Uncertain Result")
        st.warning(f"The model is only {confidence:.2f}% sure. Please ensure the image is a clear, single plant leaf and try again.")
    else:
        # Display positive result
        clean_name = predicted_class.replace("___", ": ").replace("_", " ")
        
        st.subheader(f"Diagnosis: **{clean_name}**")
        st.caption(f"Confidence: **{confidence:.2f}%**")
        
        if "healthy" in predicted_class.lower():
            st.balloons()
            st.success("✅ The plant appears healthy! Keep monitoring it.")
        else:
            st.error("🚨 DISEASE DETECTED!")
            st.info("Immediate action is recommended. Please isolate the plant and seek specific treatment based on the diagnosis.")
