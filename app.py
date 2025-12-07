import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant ID Pro",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------------
# 2. THE "NUCLEAR" CSS FIX (Forces consistent look everywhere)
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* 1. Force Light Theme & Background on EVERYTHING */
    [data-testid="stAppViewContainer"], .stApp, header, footer, .block-container {
        background-color: #F5F5F7 !important; /* Apple Light Grey */
        color: #1D1D1F !important; /* Force Dark Text */
    }
    
    /* 2. Remove top colored bar */
    header[data-testid="stHeader"] {
        background-color: #F5F5F7 !important;
        visibility: hidden;
    }

    /* 3. Center the App on Desktop (Phone View on Laptop) */
    .block-container {
        max-width: 600px;
        padding-top: 2rem;
        padding-bottom: 5rem;
        margin: auto;
    }

    /* 4. Fonts */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* 5. Headings */
    h1, h2, h3 {
        color: #1D1D1F !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    /* 6. Result Card (White floating box) */
    .result-card {
        background-color: white !important;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 2rem;
        color: #1D1D1F !important;
    }

    /* 7. Buttons (Apple Blue Pills) */
    div.stButton > button {
        background-color: #0071E3 !important;
        color: white !important;
        border-radius: 980px;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        width: 100%;
        transition: transform 0.1s ease-in-out;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        background-color: #0077ED !important;
        transform: scale(1.02);
    }
    div.stButton > button:active {
        transform: scale(0.95);
    }

    /* 8. Tutorial Text */
    .tutorial-text {
        font-size: 14px;
        color: #86868b !important;
        margin-bottom: 5px;
    }
    
    /* 9. Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# 3. MODEL SETUP
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
# 4. UI LOGIC
# ----------------------------------------------------------------------------------

if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# HEADER
st.title("Plant Health Check")
st.markdown("### Professional Grade Disease Identification")

# TUTORIAL
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    <div style="padding: 10px;">
        <p class="tutorial-text"><strong>Step 1:</strong> Select a clear photo of a <b>single plant leaf</b>.</p>
        <p class="tutorial-text"><strong>Step 2:</strong> You can either <b>upload</b> from your gallery or use the <b>camera</b>.</p>
        <p class="tutorial-text"><strong>Step 3:</strong> The AI will analyze the leaf pattern and provide a diagnosis instantly.</p>
    </div>
    """, unsafe_allow_html=True)

st.write(" ") # Spacer

# INPUT CONTROLS
col1, col2 = st.columns(2)
source_image = None

with col1:
    st.markdown("#### 📤 Upload")
    uploaded_file = st.file_uploader("Select from library", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        source_image = Image.open(uploaded_file)
        st.session_state['camera_active'] = False 

with col2:
    st.markdown("#### 📸 Camera")
    if st.button("Activate Camera"):
        st.session_state['camera_active'] = not st.session_state['camera_active']

if st.session_state['camera_active']:
    st.write("Active Camera Feed:")
    camera_pic = st.camera_input("Snap a photo", label_visibility="hidden")
    if camera_pic:
        source_image = Image.open(camera_pic)

# ----------------------------------------------------------------------------------
# 5. PREDICTION LOGIC
# ----------------------------------------------------------------------------------

if source_image:
    st.write(" ")
    st.image(source_image, caption="Analysis Target", width=400)
    
    with st.spinner("Analyzing leaf structure..."):
        image = ImageOps.fit(source_image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    # Display Result
    display_name = predicted_class.replace("___", " • ").replace("_", " ")

    html_content = f"""
    <div class="result-card">
        <h3 style="color: #86868b !important; font-size: 14px; text-transform: uppercase;">Diagnosis</h3>
        <h1 style="margin: 10px 0; font-size: 32px; color: #1D1D1F !important;">{display_name}</h1>
        <p style="color: {'#1d1d1f' if confidence > 70 else '#ff3b30'} !important; font-weight: 500;">
            Confidence: {confidence:.1f}%
        </p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    if "healthy" in predicted_class.lower():
        st.success("Analysis Complete: No threats detected.")
    elif predicted_class == 'Background_without_leaves':
        st.warning("Analysis Inconclusive: No leaf structure found.")
    else:
        st.info("Analysis Complete: Disease markers identified. Treatment recommended.")
