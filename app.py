import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Identification Through Scanning☘️",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------------
# 2. ADVANCED CSS (Dark, Blurry, Neutral Theme)
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ONE UI INSPIRED BACKGROUND (Deep OLED Dark) --- */
    [data-testid="stAppViewContainer"] {
        /* Smoother, deeper gradient for that premium OLED look */
        background: linear-gradient(160deg, #000000, #0a0a0a, #151515);
        background-size: 200% 200%;
        animation: gradientBG 20s ease infinite;
        color: #f5f5f5;
    }
    
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* --- 2. ONE UI CARDS (The "Bubble" Look) --- */
    .glass-card {
        background-color: #252525; /* Solid, accessible dark grey surface */
        border-radius: 28px; /* Signature large rounded corners */
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05); /* Very subtle border */
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* Hover effect for interactivity */
    .glass-card:hover {
        background-color: #2a2a2a;
        transform: scale(1.005);
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    /* --- 3. CAMERA ZONE (Viewfinder Style) --- */
    .camera-zone {
        background-color: #1a1a1a !important;
        border: 2px dashed #00c864 !important;
        border-radius: 24px;
        position: relative;
    }
    .scan-header {
        color: #00c864 !important;
        font-family: sans-serif;
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        display: block;
        text-align: center;
        opacity: 0.9;
    }

    /* --- 4. TYPOGRAPHY (Clean & Spacious) --- */
    h1 {
        font-family: sans-serif;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    h2, h3, h4, h5 {
        font-family: sans-serif;
        font-weight: 600;
        color: #f0f0f0 !important;
    }
    p, label, li, .stMarkdown, .stExpander p {
        color: #b0b0b0 !important;
        font-size: 15px;
        line-height: 1.6;
    }

    /* --- 5. BUTTONS (One UI Pill Shape) --- */
    div.stButton > button {
        background-color: #3e3e3e;
        color: #ffffff !important;
        border-radius: 50px; /* Full Pill Shape */
        border: none;
        padding: 16px 32px; /* Taller, comfortable touch targets */
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: capitalize; /* Cleaner look than uppercase */
        width: 100%;
        transition: background-color 0.2s ease, transform 0.1s ease;
        box-shadow: none; /* Flat design */
    }
    
    div.stButton > button:hover {
        background-color: #505050;
        transform: translateY(-2px);
    }
    
    div.stButton > button:active {
        transform: scale(0.98);
        background-color: #303030;
    }

    /* --- 6. FILE UPLOADER (Integrated Look) --- */
    [data-testid="stFileUploader"] {
        background-color: #1e1e1e;
        border-radius: 20px;
        padding: 20px;
        border: 1px dashed #444;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #666;
    }

    /* --- 7. CLEANUP & LAYOUT --- */
    #MainMenu, header, footer {visibility: hidden;}
    
    .block-container {
        max-width: 900px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }
    
    /* Rounded Images */
    div[data-testid="stImage"] img {
        border-radius: 24px;
    }
    
    /* Expander styling to match cards */
    div[data-testid="stExpander"] {
        background-color: transparent;
        border: none;
    }
    div[data-testid="stExpander"] summary {
        color: #e0e0e0 !important;
        font-weight: 600;
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
if 'camera_active' not in st.session_state: st.session_state['camera_active'] = False
if 'source_image' not in st.session_state: st.session_state['source_image'] = None

# HEADER
st.markdown("<h1 style='text-align: center; font-size: 48px; margin-bottom: 10px;'>Plant Disease Identification Through Scanning☘️</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; opacity: 0.9; margin-top: 0; margin-bottom: 2rem;'>Agricultural Diagnostics Tool</h4>", unsafe_allow_html=True)
# --- MAIN CONTROL PANEL ---
col_L, col_main, col_R = st.columns([1, 8, 1])

with col_main:
    # Everything is now inside ONE main glass card to remove empty boxes
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # --- TUTORIAL EXPANDER ---
    with st.expander("ℹ️ How to Use This Scanner", expanded=False):
        st.markdown("""
        1. **Select Input:** Choose to **Upload** a file or open the **Camera**.
        2. **Capture Leaf:** Ensure the leaf is well-lit and centered.
        3. **Analyze:** The AI will instantly process the image and provide a diagnosis.
        """)
    
    st.write("") # Small spacer

    # Input Toggle Buttons
    tog_c1, tog_c2 = st.columns(2)
    with tog_c1:
         if st.button("📁 Upload File", help="Select an image from your device"):
             st.session_state['camera_active'] = False
    with tog_c2:
        cam_label = "📸 Close Camera" if st.session_state['camera_active'] else "📸 Open Scanner"
        if st.button(cam_label, help="Activate your device camera"):
            st.session_state['camera_active'] = not st.session_state['camera_active']
            
    # --- INPUT AREA ---
    if st.session_state['camera_active']:
        # Active Camera Zone
        st.markdown('<div class="glass-card camera-zone">', unsafe_allow_html=True)
        st.markdown('<span class="scan-header">● SYSTEM ACTIVE ●</span>', unsafe_allow_html=True)
        camera_pic = st.camera_input("Center leaf...", label_visibility="hidden")
        if camera_pic:
            st.session_state['source_image'] = Image.open(camera_pic)
            st.session_state['camera_active'] = False
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
    img = st.session_state['source_image']
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process
    processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(processed_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Processing neural network analysis..."):
        preds = model.predict(img_array)
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        conf = np.max(preds[0]) * 100

    # Layout
    res_c1, res_c2 = st.columns([2, 3], gap="medium")
    
    with res_c1:
        st.markdown('<div class="glass-card" style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
        st.image(img, caption="Scan Target", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_c2:
        # --- THEME & MESSAGE LOGIC (FIXED) ---
        is_healthy = "healthy" in label.lower()
        is_bg = label == 'Background_without_leaves'
        
        if is_healthy:
            theme_color = "#00c864" # Green
            status_icon = "✅"
            status_msg = "Plant vital signs are stable."
            display_name = label.replace("___", " • ").replace("_", " ")
        elif is_bg:
            # --- FIX FOR NON-PLANT IMAGES ---
            theme_color = "#9e9e9e" # Grey
            status_icon = "🚫"
            # Bold message as requested
            status_msg = "**Non-Plant Image Detected. Please take a plant image.**"
            display_name = "Unknown Object"
        else:
            theme_color = "#c80040" # Red
            status_icon = "🚨"
            status_msg = "Pathogen markers detected."
            display_name = label.replace("___", " • ").replace("_", " ")

        # Dynamic Result Card
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
