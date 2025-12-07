import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'plant_disease_model.h5'

# Your EXACT Class Names from Colab
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

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- HELPER FUNCTIONS ---
def clean_class_name(name):
    """Converts 'Apple___Apple_scab' to 'Apple - Apple Scab'"""
    if name == 'Background_without_leaves':
        return "Unknown / Background"
    
    # Replace the triple underscore with a clean separator
    clean_name = name.replace("___", " - ").replace("_", " ")
    return clean_name.title()

def get_treatment_info(class_name):
    # Dictionary of treatments
    treatments = {
        'Apple___Apple_scab': "Apply fungicides like captan or myclobutanil. Rake up and destroy fallen leaves to reduce spread.",
        'Potato___Early_blight': "Use copper-based fungicides. Rotate crops and ensure good air circulation.",
        'Potato___Late_blight': "⚠️ Serious! Remove infected plants immediately. Apply fungicides containing chlorothalonil.",
        'Tomato___Bacterial_spot': "Apply copper sprays. Avoid overhead watering to reduce spread.",
        'Tomato___Early_blight': "Mulch soil to prevent splashing. Prune bottom leaves to improve airflow.",
        'Corn___Common_rust': "Plant resistant varieties. Apply fungicides if infection is severe early in the season.",
        'Background_without_leaves': "Please upload a clear image of a plant leaf."
    }
    
    # Logic: If healthy, say so. If not in list, give general advice.
    if "healthy" in class_name.lower():
        return "✅ No treatment needed. Keep monitoring water and nutrients."
    
    return treatments.get(class_name, "Consult a local agricultural expert for specific treatment options for this condition.")

# --- UI DESIGN ---
st.title("🌿 Plant Disease Identifier")
st.write("Upload a photo of a plant leaf to identify diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Plant'):
        with st.spinner('Analyzing...'):
            try:
                # Preprocess
                img = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array)
                predicted_index = np.argmax(predictions[0])
                predicted_raw_name = CLASS_NAMES[predicted_index]
                confidence = 100 * np.max(predictions[0])

                # Display Results
                st.divider()
                
                # Special case for background
                if predicted_raw_name == 'Background_without_leaves':
                    st.warning("⚠️ No leaf detected. Please try a clearer image.")
                else:
                    display_name = clean_class_name(predicted_raw_name)
                    st.success(f"Result: **{display_name}**")
                    st.info(f"Confidence: {confidence:.2f}%")
                    
                    st.subheader("💡 Recommendation")
                    st.write(get_treatment_info(predicted_raw_name))

            except Exception as e:
                st.error(f"Error: {e}")
