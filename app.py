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
# 2. ADVANCED CSS (Dark, Blurry, Neutral Theme)
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ANIMATED DARK BLURRY BACKGROUND (No Blue/Purple) --- */
    [data-testid="stAppViewContainer"] {
        /* Deep charcoal, slate grey, and black gradient */
        background: linear-gradient(-45deg, #1a1a1a, #2d2d2d, #000000, #363636);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: #e0e0e0;
    }
    
    /* Add a blur effect on top of the background */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; 
        left: 0;
        width: 100%; 
        height: 100%;
        backdrop-filter: blur(15px); /* Adjustable blur amount */
        z-index: -1;
    }
    
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* --- 2. GLASSMORPHISM CARDS (Neutral Dark Tint) --- */
    .glass-card {
        background: rgba(40, 40, 40, 0.5); /* Darker, neutral tint */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.8);
    }

    /* --- 3. CAMERA ZONE (Neutral/Green for Action) --- */
    .camera-zone {
        background: rgba(0, 100, 50, 0.1) !important;
        border: 1px solid rgba(0, 200, 100, 0.2) !important;
        position: relative;
        overflow: hidden;
    }
    .scan-header {
        color: #00c864 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 15px;
        display: block;
        text-align: center;
        text-shadow: 0 0 8px rgba(0,200,100,0.4);
    }

    /* --- 4. TYPOGRAPHY (White/Light Grey) --- */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800;
        text-shadow: 0 2px 5px rgba(0,0,0,0.5); 
    }
    p, label, li, .stMarkdown, .stExpander p {
        color: #cccccc !important;
    }

    /* --- 5. BUTTONS (Neutral Dark Gradient) --- */
    div.stButton > button {
        background: linear-gradient(135deg, #3a3a3a 0%, #1a1a1a 100%);
        color: #ffffff !important;
        border-radius: 12px;
        border: none;
        padding: 14px 24px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, #4a4a4a 0%, #2a2a2a 100%);
    }

    /* --- 6. CLEANUP & LAYOUT --- */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { 
        max-width: 1000px; 
        padding-top: 3rem; 
        padding-bottom: 3rem; 
    }
    div[data-testid="stImage"] { display: block; margin: auto; border-radius: 16px; overflow: hidden; }
    
    /* Make expander headers stand out */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff !important;
    }
    /* Clean up file uploader padding */
    [data-testid="stFileUploader"] {
        padding-top: 1rem;
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
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False
if 'source_image' not in st.session_state:
    st.session_state['source_image'] = None

# HEADER
st.markdown("<h1 style='text-align: center; font-size: 48px; margin-bottom: 10px;'>Plant Disease Identification Through Scanning</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; opacity: 0.9; margin-top: 0; margin-bottom: 2rem;'>AI-Powered Agricultural Diagnostics</h4>", unsafe_allow_html=True)

# --- MAIN CONTROL PANEL ---
# Tight layout: Spacer | Main Content | Spacer
col_L, col_main, col_R = st.columns([1, 8, 1])

with col_main:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # --- TUTORIAL EXPANDER (Inside the card for a cleaner look) ---
    with st.expander("ℹ️ How to Use This Scanner", expanded=False):
        st.markdown("""
        1. **Select Input:** Choose to **Upload** a file or open the **Camera**.
        2. **Capture Leaf:** Ensure the leaf is well-lit and centered.
        3. **Analyze:** The AI will instantly process the image and provide a diagnosis.
        """)
    
    st.write("") # Small spacer between tutorial and controls

    # Input Toggle Buttons (Side by Side)
    tog_c1, tog_c2 = st.columns(2)
    with tog_c1:
        if st.button("📁 Upload File", help="Select an image from your device"):
            st.session_state['camera_active'] = False
            
    with tog_c2:
        # Highlight camera button label based on state
        cam_label = "📸 Close Camera" if st.session_state['camera_active'] else "📸 Open Scanner"
        if st.button(cam_label, help="Activate your device camera"):
            st.session_state['camera_active'] = not st.session_state['camera_active']
            
    # --- INPUT AREA ---
    # Logic: Show Camera OR Upload, not both to save space
    if st.session_state['camera_active']:
        st.write("")
        # Active Camera Zone
        st.markdown('<div class="glass-card camera-zone" style="margin-bottom: 0;">', unsafe_allow_html=True)
        st.markdown('<span class="scan-header">● SYSTEM ACTIVE ●</span>', unsafe_allow_html=True)
        
        camera_pic = st.camera_input("Center leaf...", label_visibility="hidden")
        if camera_pic:
            st.session_state['source_image'] = Image.open(camera_pic)
            st.session_state['camera_active'] = False # Auto-close
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Standard Upload
        uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"], label_visibility="hidden")
        if uploaded_file:
            st.session_state['source_image'] = Image.open(uploaded_file)

    st.markdown('</div>', unsafe_allow_html=True) # End Main Glass Card

# ----------------------------------------------------------------------------------
# 5. RESULTS SECTION
# ----------------------------------------------------------------------------------
if st.session_state['source_image']:
    
    # Process & Predict
    img = st.session_state['source_image']
    processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(processed_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Processing neural network analysis..."):
        preds = model.predict(img_array)
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        conf = np.max(preds[0]) * 100

    # Layout: Image Left | Report Right (On Desktop)
    st.write("") # Spacer before results
    col_res_L, col_res_R = st.columns([2, 3], gap="medium")
    
    with col_res_L:
        st.markdown('<div class="glass-card" style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
        st.image(img, caption="Scan Target", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_res_R:
        # Theme Logic
        is_healthy = "healthy" in label.lower()
        is_bg = label == 'Background_without_leaves'
        
        if is_healthy:
            theme_color = "#00c864" # Neutral Green
            status_icon = "✅"
            status_msg = "Plant vital signs are stable."
        elif is_bg:
            theme_color = "#c8a000" # Neutral Yellow/Gold
            status_icon = "⚠️"
            status_msg = "Scan inconclusive. No clear leaf found."
        else:
            theme_color = "#c80040" # Neutral Red
            status_icon = "🚨"
            status_msg = "Pathogen markers detected."

        display_name = label.replace("___", " • ").replace("_", " ")

        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {theme_color}; height: 100%;">
            <h5 style="color: {theme_color} !important; margin:0; letter-spacing: 2px;">DIAGNOSIS</h5>
            <h2 style="font-size: 36px; margin: 15px 0; text-transform: capitalize;">{display_name}</h2>
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <span style="font-size: 24px; margin-right: 10px;">{status_icon}</span>
                <p style="font-size: 18px; margin: 0; color: #ffffff !important;">Confidence: <b>{conf:.1f}%</b></p>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <p style="font-size: 16px; line-height: 1.6; color: #cccccc;">{status_msg}</p>
        </div>
        """, unsafe_allow_html=True)
