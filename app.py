import streamlit as st
from PIL import Image
import cv2
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Lightweight CV + NLP Demo",
    layout="wide",
    page_icon="📸"
)

# --- MODEL LOADING & CACHING ---
@st.cache_resource
def load_model_and_encoder():
    try:
        # Ganti dengan path model Anda yang sebenarnya
        svm_model = joblib.load("models/svm_lightweight_model.pkl")
        le = joblib.load("models/label_encoder.pkl")
        return svm_model, le
    except FileNotFoundError:
        st.warning("Model files not found. Prediction will use placeholders.")
        # Dummy classes for demonstration if files are missing
        class DummyModel:
            def predict(self, features):
                # Simulate a prediction (e.g., class index 0)
                return np.array([0]) 
        class DummyEncoder:
            def inverse_transform(self, pred):
                # Simulate a detected label
                return ["person"] 
        return DummyModel(), DummyEncoder()

svm_model, le = load_model_and_encoder()

# --- PREPROCESSING & FEATURE EXTRACTION (No changes) ---
def preprocess(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_proc = clahe.apply(img_gray)
    return img_proc

def extract_features(img):
    orb = cv2.ORB_create(nfeatures=100)
    kp, des = orb.detectAndCompute(img, None)
    if des is None: des = np.zeros((1,32), dtype=np.uint8)
    flat = des.flatten()
    if flat.shape[0] < 3200: flat = np.pad(flat, (0, 3200 - flat.shape[0]))
    if flat.shape[0] > 3200: flat = flat[:3200]
    return flat

# --- LLM SIMULATION FUNCTION ---
def generate_llm_caption(labels):
    """Fungsi Placeholder untuk meniru output dari Model LLM Caption Generator"""
    
    # 1. Konversi labels list menjadi string
    label_list = [str(label).capitalize() for label in labels]
    
    # 2. Logika Template Sederhana
    if "person" in labels and "car" in labels:
        caption = "A person is standing next to a car parked on the street."
    elif len(labels) > 1:
        caption = f"The image features several objects including {', '.join(label_list[:-1])} and a {label_list[-1]}."
    else:
        caption = f"A single {label_list[0]} is clearly visible in the foreground."
        
    return caption

# ----------------------------------------------------
# APLIKASI UTAMA STREAMLIT
# ----------------------------------------------------

st.title("📸 Lightweight CV + NLP Demo")
st.markdown("---")

# 1. Upload Image (Di kolom Kiri)
uploaded_file = st.file_uploader("Upload Image untuk Analisis", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Menggunakan st.columns untuk tata letak yang lebih baik
    col_img, col_results = st.columns([1, 1.5]) 

    # --- Kolom Kiri: Display Gambar ---
    with col_img:
        st.subheader("1. Gambar Diunggah")
        
        st.image(img, caption="Uploaded Image", use_container_width=False)
        
        st.markdown("---")
        
        # 2. Tombol Generate Caption (Pemicu)
        if st.button("▶️ Generate Caption", type="primary"):
            st.session_state['run_analysis'] = True
            
        if 'run_analysis' not in st.session_state:
             st.session_state['run_analysis'] = False

    # --- Kolom Kanan: Hasil Analisis ---
    with col_results:
        st.subheader("2. Hasil Analisis")

        # Inisialisasi placeholder
        placeholder_cv = st.empty()
        placeholder_llm = st.empty()
        
        placeholder_cv.info("Tekan tombol 'Generate Caption' untuk memulai analisis.")
        placeholder_llm.markdown("*(Menunggu hasil CV...)*")

        # 3. Logika Analisis (Terpicu oleh Tombol)
        if st.session_state.run_analysis:
            
            # --- Tahap CV (Object Detection/Klasifikasi) ---
            with st.spinner('1/2. Running Computer Vision (CV) Model...'):
                img_proc = preprocess(img)
                features = extract_features(img_proc)
                features_full = np.concatenate([features, np.zeros(8), np.zeros(512)]).reshape(1,-1)
                
                # Prediction
                pred = svm_model.predict(features_full)
                labels = le.inverse_transform(pred)
                labels_str = ', '.join([str(label) for label in labels])

            # 3. Kasi placeholder object ke detect apa (Output CV)
            placeholder_cv.success(f"✅ CV Model Detected Objects:")
            st.markdown(f"**Objek Terdeteksi:** `{labels_str}`")

            # --- Tahap NLP (Caption Generation) ---
            with st.spinner('2/2. Running NLP/LLM Caption Generator...'):
                # 4. Kasi text buat munculin caption dr LLM nya
                llm_caption = generate_llm_caption(labels)

            placeholder_llm.success(f"🤖 LLM Generated Caption:")
            st.markdown(f"**Caption:** *{llm_caption}*")

            st.session_state.run_analysis = False # Reset state