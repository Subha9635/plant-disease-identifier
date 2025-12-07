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
# 2. ADVANCED CSS (Animated Dark Glass)
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ANIMATED DARK GRADIENT BACKGROUND --- */
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    [data-testid="stAppViewContainer"] {
        /* Deep space colors: Dark Blue -> Purple -> Deep Black */
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141E30);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #FFFFFF; /* Ensure main text color is white */
    }
    
    /* --- 2. PREMIUM GLASSMORPHISM CARDS --- */
    .glass-card {
        /* Frosted Glass Effect */
        background: rgba(255, 255, 255, 0.08); /* Slight white tint */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5); /* Deep shadow for pop */
        backdrop-filter: blur(12px); /* Stronger blur */
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.15); /* Bright thin border */
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 15px 45px 0 rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.3); /* Brighter border on hover */
    }

    /* --- 3. SPECIAL "CAMERA ZONE" STYLING --- */
    .camera-zone {
        background: rgba(0, 255, 136, 0.05) !important; /* Subtle green tint */
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        position: relative;
        overflow: hidden;
    }
    .camera-zone::before {
        /* A subtle scanning line animation */
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.2), transparent);
        animation: scan 3s linear infinite;
    }
    @keyframes scan {
        0% {left: -100%;}
        100% {left: 100%;}
    }
    .scan-header {
        color: #00ff88 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 15px;
        display: block;
        text-align: center;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }

    /* --- 4. TYPOGRAPHY & READABILITY --- */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 800;
        letter-spacing: -0.5px;
        /* Text shadow ensures readability against moving background */
        text-shadow: 0 2px 10px rgba(0,0,0,0.8); 
    }
    p, label, .stMarkdown {
        color: #E0E0E0 !important; /* Slightly off-white for body text */
    }

    /* --- 5. NEON BUTTONS --- */
    div.stButton > button {
        /* Neon Blue to Cyan Gradient */
        background: linear-gradient(135deg, #0061ff 0%, #60efff 100%);
        color: #000000 !important; /* Black text for max contrast */
        border-radius: 16px;
        border: none;
        padding: 14px 28px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 25px rgba(0, 97, 255, 0.4); /* Glowing shadow */
    }
    
    div.stButton > button:hover {
        transform: scale(1.03) translateY(-2px);
        box-shadow: 0 8px 35px rgba(96, 239, 255, 0.6);
        background: linear-gradient(135deg, #60efff 0%, #0061ff 100%); /* Invert on hover */
    }

    /* --- 6. CLEANUP --- */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { max-width: 1100px; padding-top: 3rem; padding-bottom: 5rem; }
    div[data-testid="stImage"] { display: block; margin: auto; border-radius: 16px; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# 3. MODEL SETUP
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Ensure this matches your file name exactly
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
# New Title
st.markdown("<h1 style='text-align: center; font-size: 48px;'>Plant Disease Identification Through Scanning</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; opacity: 0.8;'>AI-Powered Agricultural Diagnostics</h3>", unsafe_allow_html=True)

st.write(" ")

# --- MAIN CONTROL PANEL ---
# Using columns to center the input section
col_spacer_l, col_main, col_spacer_r = st.columns([1, 5, 1])

with col_main:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Input Toggle Buttons
    tog_c1, tog_c2 = st.columns(2)
    with tog_c1:
        if st.button("📁 Upload File", help="Select an image from your device"):
            st.session_state['camera_active'] = False
    with tog_c2:
        # Highlight camera button if active
        cam_label = "📸 Close Camera" if st.session_state['camera_active'] else "📸 Open Scanner"
        if st.button(cam_label, help="Activate your device camera"):
            st.session_state['camera_active'] = not st.session_state['camera_active']
            
    st.write(" ") # Spacer

    # --- INPUT AREA (Lively Camera Section) ---
    if st.session_state['camera_active']:
        # Wrap camera in the special glowing "camera-zone" card
        st.markdown('<div class="glass-card camera-zone">', unsafe_allow_html=True)
        st.markdown('<span class="scan-header">● ACTIVE SCANNING MODULE INITIATED ●</span>', unsafe_allow_html=True)
        
        camera_pic = st.camera_input("Center leaf in view...", label_visibility="hidden")
        if camera_pic:
            st.session_state['source_image'] = Image.open(camera_pic)
            st.session_state['camera_active'] = False # Auto-close after taking
            st.experimental_rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Standard upload section
        uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"], label_visibility="hidden")
        if uploaded_file:
            st.session_state['source_image'] = Image.open(uploaded_file)

    st.markdown('</div>', unsafe_allow_html=True) # End Main Glass Card

# ----------------------------------------------------------------------------------
# 5. RESULTS SECTION
# ----------------------------------------------------------------------------------
if st.session_state['source_image']:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Split layout: Image Left | Report Right
    res_c1, res_c2 = st.columns([2, 3])
    
    with res_c1:
        # Display Image in a tight glass container
        st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
        st.image(st.session_state['source_image'], use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_c2:
        with st.spinner("Processing neural network analysis..."):
            # Process & Predict
            img = st.session_state['source_image']
            processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(processed_img)
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            label = CLASS_NAMES[idx]
            conf = np.max(preds[0]) * 100

        # Define theme colors based on result
        is_healthy = "healthy" in label.lower()
        is_bg = label == 'Background_without_leaves'
        
        if is_healthy:
            theme_color = "#00ff88" # Neon Green
            status_icon = "✅"
            status_msg = "Plant vital signs are stable. No pathogens detected."
        elif is_bg:
            theme_color = "#ffcc00" # Neon Yellow
            status_icon = "⚠️"
            status_msg = "Scan inconclusive. No clear leaf structure identified."
        else:
            theme_color = "#ff0055" # Neon Red/Pink
            status_icon = "🚨"
            status_msg = "Pathogen markers detected. Recommended action required."

        display_name = label.replace("___", " • ").replace("_", " ")

        # Dynamic Result Glass Card
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {theme_color}; box-shadow: 0 10px 30px -10px {theme_color};">
            <h4 style="color: {theme_color} !important; margin:0; letter-spacing: 1px;">ANALYSIS REPORT</h4>
            <h2 style="font-size: 36px; margin: 15px 0; text-transform: capitalize;">
                {display_name}
            </h2>
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <span style="font-size: 24px; margin-right: 10px;">{status_icon}</span>
                <p style="font-size: 18px; margin: 0;">AI Confidence: <b>{conf:.1f}%</b></p>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="font-size: 16px; line-height: 1.5; color: #e0e0e0;">
                {status_msg}
            </p>
        </div>
        """, unsafe_allow_html=True)
