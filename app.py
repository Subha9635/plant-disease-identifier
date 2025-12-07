import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Identification Through Scanning",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------------
# 2. MODERN GLASSMORPHISM CSS
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ANIMATED BACKGROUND --- */
    @keyframes gradientAnimation {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #2b2b2b, #000000, #1a1a1a, #242424);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        color: #FFFFFF;
    }
    
    /* --- 2. GLASS CARDS (For Results & Containers) --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.05); /* Very subtle white tint */
        backdrop-filter: blur(20px); /* Stronger blur for premium feel */
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* --- 3. TYPOGRAPHY --- */
    h1, h2, h3, h4, h5 {
        color: #FFFFFF !important;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    p, div, label, .stMarkdown {
        color: #E0E0E0 !important;
    }

    /* --- 4. BUTTONS (Clean Glassy Look) --- */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 15px 25px;
        font-size: 16px;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.4);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }

    /* --- 5. HIDE DEFAULT ELEMENTS --- */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 5rem; 
        max-width: 1100px;
    }
    
    /* File Uploader Customization */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px dashed rgba(255, 255, 255, 0.2);
    }
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
# 4. UI LAYOUT
# ----------------------------------------------------------------------------------

# State
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# Header
col_head_1, col_head_2 = st.columns([1, 4])
with col_head_2:
    st.title("Plant ID Pro")
    st.markdown("#### AI-Powered Disease Diagnostics")

# Spacer
st.markdown("<br>", unsafe_allow_html=True)

# How To Use Section
with st.expander("ℹ️ How to Use This Scanner"):
    st.markdown("""
    1. **Upload or Scan:** Select an image from your device or use the camera.
    2. **Analyze:** The AI will automatically detect the plant health status.
    3. **Results:** View the diagnosis and confidence score below.
    """)

# Main Content Area
col_spacer_L, col_main, col_spacer_R = st.columns([1, 6, 1])

with col_main:
    # --- CONTROLS (No empty wrapper box now) ---
    input_c1, input_c2 = st.columns(2)
    
    with input_c1:
        st.markdown("##### 📁 Upload Image")
        # Standard File Uploader (Styled by CSS)
        uploaded_file = st.file_uploader("Select", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state['source_image'] = Image.open(uploaded_file)
            st.session_state['camera_active'] = False

    with input_c2:
        st.markdown("##### 📸 Use Camera")
        if st.button("Toggle Camera"):
            st.session_state['camera_active'] = not st.session_state['camera_active']
            
    # Camera Logic
    if st.session_state['camera_active']:
        st.markdown("---")
        camera_pic = st.camera_input("Take a picture", label_visibility="collapsed")
        if camera_pic:
            st.session_state['source_image'] = Image.open(camera_pic)

# ----------------------------------------------------------------------------------
# 5. RESULTS DISPLAY
# ----------------------------------------------------------------------------------

if 'source_image' in st.session_state and st.session_state['source_image']:
    img = st.session_state['source_image']
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process
    processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(processed_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing..."):
        preds = model.predict(img_array)
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        conf = np.max(preds[0]) * 100

    # Layout for Results
    res_c1, res_c2 = st.columns([1, 1])
    
    with res_c1:
        # Image inside a glass card
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.image(img, caption="Analyzed Leaf", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_c2:
        # Dynamic Result Card
        status_color = "#00ff88" if "healthy" in label.lower() else "#ff4b4b"
        display_name = label.replace("___", " • ").replace("_", " ")

        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid {status_color};">
            <h4 style="color: #cccccc; margin:0; text-transform: uppercase; font-size: 14px;">Diagnosis Report</h4>
            <h1 style="font-size: 38px; margin: 10px 0; background: -webkit-linear-gradient(45deg, #fff, {status_color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {display_name}
            </h1>
            <p style="font-size: 18px; color: #E0E0E0;">Confidence: <b>{conf:.1f}%</b></p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="color: #E0E0E0;">
                {"✅ This plant looks vibrant and healthy." if "healthy" in label.lower() 
                else "⚠️ Disease markers detected. Isolate plant and consult treatment."}
            </p>
        </div>
        """, unsafe_allow_html=True)
