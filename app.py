import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------------
# 1. CONFIGURATION & STYLE (The "Apple" Look)
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant ID Pro",
    page_icon="🌿",
    layout="centered", # Centered looks more "App-like" on desktop
    initial_sidebar_state="collapsed"
)

# Custom CSS for Apple-like Aesthetics
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #F5F5F7; /* Apple Light Grey */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1D1D1F;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Modern Card Style for Results */
    .result-card {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 2rem;
    }
    
    /* Hide the default Streamlit main menu and footer for cleanliness */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Button Styling (Apple Blue Pills) */
    div.stButton > button {
        background-color: #0071E3;
        color: white;
        border-radius: 980px; /* Pill shape */
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #0077ED;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# 2. MODEL SETUP
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_model.h5')

model = load_model()

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

# ----------------------------------------------------------------------------------
# 3. UI LOGIC (Camera Toggle)
# ----------------------------------------------------------------------------------

# Initialize session state for camera visibility
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# Header
st.title("Plant Health Check")
st.markdown("### Professional Grade Disease Identification")
st.write(" ") # Spacer

# --- CONTROLS ---
# We use columns to center the buttons or layout nicely
col1, col2 = st.columns(2)

source_image = None

with col1:
    st.markdown("#### 📤 Upload Photo")
    uploaded_file = st.file_uploader("Select from library", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        source_image = Image.open(uploaded_file)
        # Turn off camera if upload is used
        st.session_state['camera_active'] = False 

with col2:
    st.markdown("#### 📸 Take Picture")
    # Toggle button
    if st.button("Activate Camera"):
        st.session_state['camera_active'] = not st.session_state['camera_active']

# --- CAMERA INPUT SECTION ---
# Only show the camera widget if the state is True
if st.session_state['camera_active']:
    st.write("Context: Active Camera Feed")
    camera_pic = st.camera_input("Snap a photo", label_visibility="hidden")
    if camera_pic:
        source_image = Image.open(camera_pic)

# ----------------------------------------------------------------------------------
# 4. PREDICTION & RESULTS
# ----------------------------------------------------------------------------------

if source_image:
    # Display the image cleanly
    st.image(source_image, caption="Analysis Target", width=400)
    
    # Progress Bar for "Scanning" effect
    with st.spinner("Analyzing leaf structure..."):
        # Preprocess
        image = ImageOps.fit(source_image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    # --- THE "MODERN CARD" RESULT DISPLAY ---
    
    # Clean up the name (Remove underscores)
    display_name = predicted_class.replace("___", " • ").replace("_", " ")

    # Custom HTML for the Result Card
    html_content = f"""
    <div class="result-card">
        <h3 style="color: #86868b; font-size: 14px; text-transform: uppercase;">Diagnosis</h3>
        <h1 style="margin: 10px 0; font-size: 32px;">{display_name}</h1>
        <p style="color: {'#1d1d1f' if confidence > 70 else '#ff3b30'}; font-weight: 500;">
            Confidence: {confidence:.1f}%
        </p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    # Treatment Section (Optional polish)
    if "healthy" in predicted_class.lower():
        st.success("Analysis Complete: No threats detected.")
    elif predicted_class == 'Background_without_leaves':
        st.warning("Analysis Inconclusive: No leaf structure found.")
    else:
        st.info("Analysis Complete: Disease markers identified. Treatment recommended.")
