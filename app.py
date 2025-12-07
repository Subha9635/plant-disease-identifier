import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Identification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------------
# 2. ULTRA-GLASS ANIMATED CSS
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ANIMATED LIVING BACKGROUND --- */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stAppViewContainer"] {
        /* Deep space/ocean gradient */
        background: linear-gradient(-45deg, #020024, #090979, #00d4ff, #090979);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: #FFFFFF;
    }
    
    /* Overlay to darken the background slightly so text pops */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }
    
    /* --- 2. RESPONSIVE CONTAINER --- */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
        max-width: 1100px;
        margin: auto;
    }

    /* --- 3. ENHANCED GLASSMORPHISM CARDS --- */
    .glass-card {
        /* More translucent background */
        background: rgba(255, 255, 255, 0.03);
        /* Stronger blur for icy effect */
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        /* Brighter, crisper border */
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(96, 239, 255, 0.1); /* Subtle blue glow on hover */
    }

    /* Special styling for the large camera section */
    .camera-focus-mode {
        text-align: center;
        border: 2px solid rgba(96, 239, 255, 0.3); /* Highlighted border */
    }

    /* --- 4. TYPOGRAPHY & ACCENTS --- */
    h1, h2, h3, h4, h5 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    
    /* Main Title Gradient effect */
    .main-title {
        background: -webkit-linear-gradient(0deg, #ffffff, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }

    p, label, li {
        color: #E0E0E0 !important;
        font-size: 1.1rem;
    }

    /* --- 5. NEON BUTTONS --- */
    div.stButton > button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: #FFFFFF !important;
        border-radius: 12px;
        border: none;
        padding: 16px 28px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
    }
    
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 198, 255, 0.5);
    }

    /* --- 6. HIDE JUNK --- */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Center images within cards */
    div[data-testid="stImage"] > img {
        border-radius: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# 3. MODEL SETUP
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Ensure this file is in your folder
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

# Initialize State
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False
if 'source_image' not in st.session_state:
    st.session_state['source_image'] = None

# --- HEADER ---
col_spacer_L, col_content, col_spacer_R = st.columns([1, 8, 1])
with col_content:
    st.markdown('<h1 class="main-title">Plant Disease Identification Through Scanning</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Diagnostics Panel")

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN INPUT AREA ---
# Using 3 columns to center content on wide screens
c_L, c_Main, c_R = st.columns([1, 6, 1])

with c_Main:
    # 1. Top Controls (Side by Side)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    input_c1, input_c2 = st.columns(2)
    
    with input_c1:
        st.markdown("##### 📁 Option 1: Upload File")
        uploaded_file = st.file_uploader("Select image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state['source_image'] = Image.open(uploaded_file)
            st.session_state['camera_active'] = False # Reset camera if upload is used

    with input_c2:
        st.markdown("##### 📸 Option 2: Live Scan")
        # The toggle button lets the user choose to open the big camera view
        if st.button("Open Camera Scanner"):
            st.session_state['camera_active'] = not st.session_state['camera_active']
            # Clear previous image when opening camera to avoid confusion
            if st.session_state['camera_active']:
                 st.session_state['source_image'] = None
            
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. THE NEW LARGE CAMERA SECTION (Appears below buttons when active)
    if st.session_state['camera_active']:
        # We use a separate, highlighted glass card for the camera "mode"
        st.markdown('<div class="glass-card camera-focus-mode">', unsafe_allow_html=True)
        st.markdown("#### 🟢 Scanner Active: Center Leaf in Frame")
        # This will be much larger now because it's in a main column, not a side column
        camera_pic = st.camera_input("Take photo", label_visibility="collapsed")
        if camera_pic:
            st.session_state['source_image'] = Image.open(camera_pic)
            # Optional: Close camera automatically after taking pic
            # st.session_state['camera_active'] = False 
            # st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------
# 5. PREDICTION & RESULTS (Split View)
# ----------------------------------------------------------------------------------

if st.session_state['source_image']:
    img = st.session_state['source_image']
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # Split layout: Left for Image, Right for Data
    res_c1, res_c2 = st.columns([2, 3]) 
    
    with res_c1:
        st.markdown('<div class="glass-card" style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        st.markdown("##### Analyzed Sample")
        st.image(img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_c2:
        with st.spinner("Processing neural network..."):
            # Predict
            processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(processed_img)
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            label = CLASS_NAMES[idx]
            conf = np.max(preds[0]) * 100

        # Color Logic
        is_healthy = "healthy" in label.lower()
        status_color = "#00ff88" if is_healthy else "#ff3333"
        
        # Clean Name
        display_name = label.replace("___", " • ").replace("_", " ").title()

        # Result Card HTML
        st.markdown(f"""
        <div class="glass-card" style="border-left: 8px solid {status_color};">
            <h3 style="color: #b0b0b0; margin-bottom:0;">DIAGNOSIS RESULT</h3>
            <h1 style="font-size: 48px; margin: 10px 0; background: -webkit-linear-gradient(0deg, #fff, {status_color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {display_name}
            </h1>
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <span style="font-size: 24px; margin-right: 10px;">AI Confidence:</span>
                <span style="font-size: 24px; font-weight: bold; color: {status_color};">{conf:.1f}%</span>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <div style="font-size: 18px; line-height: 1.6;">
                {"✅ <b>Status: Healthy.</b> Plant shows vigorous condition with no pathogenic signs." if is_healthy
                else "⚠️ <b>Status: Infected.</b> Pathogen markers identified. Isolate this plant immediately to prevent spread."}
            </div>
        </div>
        """, unsafe_allow_html=True)
