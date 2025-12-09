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
# 2. ONE UI AESTHETIC CSS (Fixed Buttons & Animation)
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ANIMATED DARK BACKGROUND (Enhanced for All Platforms) --- */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #111111, #000000, #1a1a1a);
        background-size: 300% 300%;
        animation: gradientBG 15s ease infinite;
        color: #f5f5f5;
        /* Ensure compatibility across platforms */
        -webkit-animation: gradientBG 15s ease infinite;
        -moz-animation: gradientBG 15s ease infinite;
        -o-animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Vendor prefixes for broader support */
    @-webkit-keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    @-moz-keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    @-o-keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* --- 2. UNIVERSAL BUTTON FIX (Forces Dark Grey on Laptop & Mobile) --- */
    div.stButton > button {
        background: linear-gradient(135deg, #333333 0%, #222222 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 50px !important; /* One UI Pill Shape */
        padding: 16px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease-in-out !important;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    /* Hover State with Glow */
    div.stButton > button:hover {
        background: linear-gradient(135deg, #444444 0%, #333333 100%) !important;
        transform: scale(1.02);
        border-color: rgba(255,255,255,0.3) !important;
        box-shadow: 0 6px 20px rgba(255,255,255,0.1) !important;
    }
    
    /* Active/Click State */
    div.stButton > button:active {
        transform: scale(0.98);
        background: #111111 !important;
    }
    
    /* Subtle Button Animation */
    div.stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    div.stButton > button:hover::before {
        left: 100%;
    }

    /* --- 3. GLASS CARDS (One UI "Bubbles" with Fade-In Animation) --- */
    .glass-card {
        background-color: rgba(30, 30, 30, 0.6);
        border-radius: 26px;
        padding: 30px;
        margin-bottom: 20px;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    /* Fade-in animation for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Subtle glow on hover for cards */
    .glass-card:hover {
        box-shadow: 0 12px 40px 0 rgba(255, 255, 255, 0.1);
        transform: translateY(-5px);
        transition: all 0.3s ease;
    }

    /* --- 4. TYPOGRAPHY with Text Shadows --- */
    h1 { 
        font-weight: 800; 
        letter-spacing: -1px; 
        margin-bottom: 10px; 
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        animation: textGlow 2s ease-in-out infinite alternate;
    }
    h2, h3 { 
        font-weight: 700; 
        color: #f0f0f0 !important; 
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    p, label, li { 
        color: #cccccc !important; 
        font-size: 15px; 
        line-height: 1.6; 
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Text glow animation for header */
    @keyframes textGlow {
        from { text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
        to { text-shadow: 0 2px 10px rgba(255,255,255,0.2), 0 0 20px rgba(255,255,255,0.1); }
    }

    /* --- 5. CLEANUP (Removes extra spacing) --- */
    .block-container { 
        padding-top: 3rem; 
        padding-bottom: 5rem; 
        max-width: 900px; /* Limits width on laptop so it doesn't stretch */
    }
    
    /* Hide Header/Footer */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Center Images with Subtle Animation */
    div[data-testid="stImage"] { 
        display: block; 
        margin: auto; 
        animation: imageFadeIn 1s ease-out;
    }
    div[data-testid="stImage"] img { 
        border-radius: 20px; 
        transition: transform 0.3s ease;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.05);
    }
    
    /* Image fade-in */
    @keyframes imageFadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* File Uploader Style with Animation */
    [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 15px;
        transition: background-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        background-color: rgba(255,255,255,0.1);
    }
    
    /* --- 6. ADDITIONAL AESTHETIC ELEMENTS --- */
    /* Floating particles effect (subtle) */
    .floating-particles {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        overflow: hidden;
    }
    .particle {
        position: absolute;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
        animation: float 10s infinite linear;
    }
    .particle:nth-child(1) { width: 4px; height: 4px; top: 10%; left: 10%; animation-delay: 0s; }
    .particle:nth-child(2) { width: 6px; height: 6px; top: 20%; left: 80%; animation-delay: 2s; }
    .particle:nth-child(3) { width: 3px; height: 3px; top: 70%; left: 20%; animation-delay: 4s; }
    .particle:nth-child(4) { width: 5px; height: 5px; top: 50%; left: 60%; animation-delay: 6s; }
    .particle:nth-child(5) { width: 4px; height: 4px; top: 90%; left: 90%; animation-delay: 8s; }
    
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); opacity: 0.5; }
        50% { opacity: 1; }
        100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
    }
    
    /* Add particles to the main container */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="1" fill="rgba(255,255,255,0.1)"/></svg>') repeat;
        animation: particleMove 20s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes particleMove {
        0% { transform: translateY(0); }
        100% { transform: translateY(-100px); }
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
if 'source_image' not in st.session_state:
    st.session_state['source_image'] = None

# HEADER
st.markdown("<h1 style='text-align: center; font-size: 42px;'>Plant Disease Identification Through Scanning</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-top: -10px;'>Agricultural Diagnostics Tool☘️</p>", unsafe_allow_html=True)

# HOW TO USE (Clean Expander)
with st.expander("ℹ️ How to Use This Scanner"):
    st.markdown("""
    1. **Upload or Scan:** Select an image or use the camera.
    2. **Analyze:** The AI will automatically detect the disease.
    3. **Results:** View diagnosis and confidence score below.
    """)

# MAIN CONTROLS (Single Glass Card Container)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# Two columns for buttons (Side by Side)
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📁 Upload Image"):
        st.session_state['camera_active'] = False
        st.session_state['source_image'] = None # Reset previous image

with col_btn2:
    # Toggle Camera Button
    cam_label = "📸 Close Camera" if st.session_state['camera_active'] else "📸 Open Camera"
    if st.button(cam_label):
        st.session_state['camera_active'] = not st.session_state['camera_active']
        st.session_state['source_image'] = None # Reset previous image

# INPUT LOGIC (Clean flow, no extra containers)
if st.session_state['camera_active']:
    st.markdown("---")
    camera_pic = st.camera_input("Take a picture", label_visibility="collapsed")
    if camera_pic:
        st.session_state['source_image'] = Image.open(camera_pic)
        st.session_state['camera_active'] = False # Auto-close
        st.rerun()

elif not st.session_state['camera_active']:
    # Show uploader if camera is NOT active
    uploaded_file = st.file_uploader("Select Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state['source_image'] = Image.open(uploaded_file)

st.markdown('</div>', unsafe_allow_html=True) # End Main Card

# ----------------------------------------------------------------------------------
# 5. RESULTS DISPLAY
# ----------------------------------------------------------------------------------

if st.session_state['source_image']:
    img = st.session_state['source_image']
    
    # Analyze
    processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(processed_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing..."):
        preds = model.predict(img_array)
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        conf = np.max(preds[0]) * 100

    # Layout: Image on Top, Result Below (Better for One UI Flow)
    st.image(img, caption="Analyzed Leaf", use_column_width=False, width=350)
    
    # Logic
    is_healthy = "healthy" in label.lower()
    is_bg = label == 'Background_without_leaves'
    
    if is_healthy:
        color = "#00c864" # Green
        icon = "✅"
        msg = "Plant is healthy."
        name = label.replace("___", " • ").replace("_", " ")
    elif is_bg:
        color = "#888888" # Grey
        icon = "🚫"
        msg = "**Non-Plant Image Detected.** Please scan a valid plant leaf."
        name = "Unknown Object"
    else:
        color = "#ff4b4b" # Red
        icon = "🚨"
        msg = "Disease detected. Isolate plant."
        name = label.replace("___", " • ").replace("_", " ")

    # Result Card
    st.markdown(f"""
    <div class="glass-card" style="border-left: 6px solid {color}; text-align: left;">
        <h4 style="color: {color} !important; margin:0; font-size: 14px; text-transform: uppercase;">Diagnosis</h4>
        <h2 style="font-size: 32px; margin: 10px 0; color: white !important;">{name}</h2>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
            <p style="font-size: 18px; margin: 0; color: #eee !important;">Confidence: <b>{conf:.1f}%</b></p>
        </div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
        <p style="font-size: 16px; color: #ccc;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)
