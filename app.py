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
    layout="wide", # 🟢 CRITICAL: Switches to wide mode for Laptop use
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------------
# 2. DYNAMIC "GLASS DARK" CSS
# ----------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- 1. ANIMATED BACKGROUND --- */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% -20%, #2b2b2b, #000000);
        background-attachment: fixed;
        color: #FFFFFF;
    }
    
    /* --- 2. RESPONSIVE CONTAINER (The "Dynamic Spacing") --- */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px; /* Wider on desktop */
        margin: auto;
    }
    
    /* Mobile Override: Keep it tight on phones */
    @media (max-width: 768px) {
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    /* --- 3. GLASSMORPHISM CARDS --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.05); /* 5% White */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* --- 4. TEXT & HEADINGS --- */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    p, label {
        color: #E0E0E0 !important;
    }

    /* --- 5. BUTTONS (Neon Blue Glow) --- */
    div.stButton > button {
        background: linear-gradient(90deg, #0061ff 0%, #60efff 100%);
        color: #000000 !important; /* Black text for contrast */
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 97, 255, 0.3);
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 97, 255, 0.5);
        color: #000000 !important;
    }

    /* --- 6. HIDE JUNK --- */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Center images */
    div[data-testid="stImage"] {
        display: block;
        margin-left: auto;
        margin-right: auto;
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
# 4. DYNAMIC UI LAYOUT
# ----------------------------------------------------------------------------------

# Initialize State
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = False

# --- HEADER SECTION ---
col_head_1, col_head_2 = st.columns([1, 4]) # Logo left, Title right
with col_head_2:
    st.title("Plant ID Pro")
    st.markdown("#### AI-Powered Disease Diagnostics")

# --- MAIN CONTROLS (Responsive Grid) ---
st.markdown("<br>", unsafe_allow_html=True) # Spacer

# Using 3 columns for desktop to center content, but full width on mobile
# Logic: Empty | Content | Empty
col_spacer_L, col_main, col_spacer_R = st.columns([1, 6, 1])

with col_main:
    # Wrap controls in a Glass Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # 2-Column Inputs INSIDE the card
    input_c1, input_c2 = st.columns(2)
    
    with input_c1:
        st.markdown("##### 📁 Upload Image")
        uploaded_file = st.file_uploader("Select", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state['source_image'] = Image.open(uploaded_file)
            st.session_state['camera_active'] = False

    with input_c2:
        st.markdown("##### 📸 Use Camera")
        if st.button("Toggle Camera"):
            st.session_state['camera_active'] = not st.session_state['camera_active']
            
    # Camera Area
    if st.session_state['camera_active']:
        st.markdown("---")
        camera_pic = st.camera_input("Take a picture", label_visibility="collapsed")
        if camera_pic:
            st.session_state['source_image'] = Image.open(camera_pic)
            
    st.markdown('</div>', unsafe_allow_html=True) # End Glass Card

# ----------------------------------------------------------------------------------
# 5. PREDICTION & RESULTS DISPLAY
# ----------------------------------------------------------------------------------

if 'source_image' in st.session_state and st.session_state['source_image']:
    img = st.session_state['source_image']
    
    # --- SPLIT LAYOUT FOR RESULTS (Desktop Friendly) ---
    # Left: Image | Right: Diagnosis
    res_c1, res_c2 = st.columns([1, 1]) 
    
    with res_c1:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.image(img, caption="Analyzed Leaf", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with res_c2:
        with st.spinner("Running AI Diagnosis..."):
            # Predict
            processed_img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(processed_img)
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            label = CLASS_NAMES[idx]
            conf = np.max(preds[0]) * 100

        # Dynamic Color Logic
        status_color = "#00ff88" if "healthy" in label.lower() else "#ff4b4b"
        display_name = label.replace("___", " • ").replace("_", " ")

        # Result Card HTML
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid {status_color};">
            <h4 style="color: #b0b0b0; margin:0;">DIAGNOSIS REPORT</h4>
            <h1 style="font-size: 42px; margin: 10px 0; background: -webkit-linear-gradient(45deg, #fff, {status_color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {display_name}
            </h1>
            <p style="font-size: 20px;">Confidence: <b>{conf:.1f}%</b></p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p>
                {"✅ This plant looks vibrant and healthy. No treatment required." if "healthy" in label.lower() 
                else "⚠️ Disease markers detected. Isolate plant and consult treatment database."}
            </p>
        </div>
        """, unsafe_allow_html=True)
