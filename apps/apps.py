import streamlit as st
from PIL import Image
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern, hog
from ultralytics import YOLO
import os
import requests
import json
import re
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Object Detection for Captioning",
    layout="wide",
    page_icon="📸"
)

# --- LOADING MODEL & CACHING ---
api_key = st.secrets["OPENROUTER_API_KEY"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svc_single59.pkl')
LE_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder(2).pkl')

@st.cache_resource
def load_models():
    try:
        svm_model = joblib.load(MODEL_PATH)
        # svm_model = joblib.load("models/svc_singlepca76.pkl")
        le = joblib.load(LE_PATH)
        yolo = YOLO("yolov8n.pt")
        scaler = joblib.load("scaler.pkl")

        return svm_model, le, yolo, scaler
    except Exception as e:
        st.error(f"fail to load: {e}")
        return None, None, None,None

svm_model, le, yolo, scaler = load_models()
# pca = joblib.load("models/pca256.joblib")


# --- FUNGSI PEMBANTU (CV) ---
def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def extract_hog_lbp_features(img_bgr, bbox):
    # Crop berdasarkan bounding box
    x, y, w, h = bbox
    crop = img_bgr[y:y+h, x:x+w]
    # Preprocessing
    crop = cv2.resize(crop, (224, 224))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    # HOG
    hog_feat = hog(
        gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys"
    )
    # return hog_feat
    #  LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return np.concatenate([hog_feat, lbp_hist])

def get_yolo_detections(img_bgr, conf_thresh=0.5):
    results = yolo(img_bgr, conf=conf_thresh, verbose=False)
    detections = []
    for r in results:
        if r.boxes:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                    "area": (x2-x1)*(y2-y1)
                })
    return detections

def draw_bbox(img_pil, bbox, label):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    text_label=str(label)
    
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text_label, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), (0, 255, 0), -1)

    cv2.putText(
        img, 
        text_label, #str
        (x, y - 10), 
        font, 
        font_scale, 
        (255, 255, 255), 
        thickness
    )
    # 5.RGB(utk show)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# MULTIPLE PREDICT
def draw_multiple_bboxes(img_pil, candidates):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    for cand in candidates:
        x, y, w, h = cand['bbox']
        label = cand['label']
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # label
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.5, 2)
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), (0, 255, 0), -1)
        
        cv2.putText(img, label, (x, y - 10), font, 0.5, (255, 255, 255), 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# PREDICT label
def predict_logic(img_bgr, svm_model, le, scaler, allowed_classes):
    # 1. Deteksi YOLO
    detections = get_yolo_detections(img_bgr, conf_thresh=0.5)
    
    candidates_with_labels = []
    for d in detections:
        if d['cls'] in allowed_classes:
            bbox = d["bbox"]
            # 2. Ekstraksi Fitur
            features = extract_hog_lbp_features(img_bgr, bbox)
            features = features.reshape(1, -1)
            
            # 3. Scaling
            if scaler is not None:
                features = scaler.transform(features)
            # features = pca.transform(features)
            # 4. Predict SVM
            pred_idx = svm_model.predict(features)[0]
            # ID asli dari Label Encoder (hasilnya akan: 1, 2, 3, atau 18)
            actual_id = le.inverse_transform([pred_idx])[0]

            label_map={
                1:"person",
                2:"bicycle",
                3:"car",
                18:"dog"
            }

            pred_label = label_map.get(actual_id, f"unknown ID : {actual_id}")
            # if label_map and pred_idx in label_map:
            #     pred_label=label_map[pred_idx]
            # else:
            #     pred_label = le.inverse_transform([pred_idx])[0]
            # pred_label = label_map.get(pred_idx, f"Unkown object ({pred_idx})")            
            candidates_with_labels.append({
                "bbox": bbox,
                "conf": d["conf"],
                "label": pred_label,
                "area": d["area"]
            })
            
    if not candidates_with_labels:
        return None, None
    # if candidates_with_labels:
    #     for cand in candidates_with_labels:
    #         st.write(f"Ditemukan: {cand['label']} (Area: {cand['area']})")  

    # 5. Pilih yang terbaik 
    best = max(candidates_with_labels, key=lambda d: d['conf'])
    return best['label'], best['bbox']

# MULTIPLE BBOX to extract feature
def predict_multiple(img_bgr, svm_model, le, scaler, allowed_classes):
    detections = get_yolo_detections(img_bgr, conf_thresh=0.5)
    candidates_with_labels = []
    
    for d in detections:
        if d['cls'] in allowed_classes:
            bbox = d["bbox"]
            features = extract_hog_lbp_features(img_bgr, bbox)
            features = features.reshape(1, -1)
            
            if scaler is not None:
                features = scaler.transform(features)
            
            pred_idx = svm_model.predict(features)[0]
            actual_id = le.inverse_transform([pred_idx])[0]

            label_map={1:"person", 2:"bicycle", 3:"car", 18:"dog"}
            pred_label = label_map.get(actual_id, f"unknown:{actual_id}")

            candidates_with_labels.append({
                "bbox": bbox,
                "conf": d["conf"],
                "label": pred_label
            })
    
    # KEMBALIKAN SEMUA LIST dr extract bbox
    return candidates_with_labels

# --- NLP SIMULATION ---
def generate_llm_caption(detected_label):
    """Fungsi untuk meniru output dari Model LLM Caption Generator, menggunakan label tunggal."""

    # 1. Konteks yang dikirim ke LLM adalah label objek yang terdeteksi
    llm_context = f"A high-resolution image featuring a prominent {detected_label}. Generate a caption focused on this object."

    messages=[
        {
        "role": "user",
        "content": (
            f"You are a social media content expert and a witty copywriter. "
            f"Given the following context, generate a catchy, engaging caption for a social media post. "
            f"The caption should be concise, attention-grabbing, and encourage interaction (like, comment, share). "
            f"Include 5-10 relevant hashtags at the end, based on the content.\n\n"
            f"Context:\n{llm_context}\n\n" # Menggunakan konteks yang dibuat dari label
            f"Output format:\n"
            f"Caption: <your caption here>\n"
            f"Hashtags: <relevant hashtags separated by spaces>"
            )
        }
    ]

    # Asumsikan api_key, requests, json, time, re, dan openrouter.ai sudah didefinisikan/di-import
    # ... (Bagian koneksi API dan penanganan error tetap sama) ...
    
    for _ in range(3):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "mistralai/mistral-small-3.1-24b-instruct:free",
                "messages": messages
            })
        )
        if response.status_code == 429:
            print("Rate limited, retrying in 5 seconds...")
            time.sleep(5)
            continue
        break
    
    result = response.json()
    if "choices" in result:
        content = result["choices"][0]["message"]["content"]
    else:
        error_msg = result.get("error", {}).get("message", "Unknown error")
        print("API returned an error:", error_msg)
        # Jika terjadi error, berikan caption fallback
        content = f"Caption: #{detected_label.replace(' ', '')}"
    
    # Parsing output
    caption_match = re.search(r'Caption:\s*"(.*?)"', content, re.DOTALL)
    hashtags_match = re.search(r'Hashtags:\s*(.*)', content, re.DOTALL)

    caption = caption_match.group(1).strip() if caption_match else ""
    hashtags = hashtags_match.group(1).strip() if hashtags_match else ""
    
    return {"caption": caption, "hashtags": hashtags}

# --- MAIN APP ---
st.title("📸 Object Detection for Captioning")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_pil = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    col_img, col_results = st.columns([1, 1.2])

    with col_img:
        st.subheader("Uploaded image")
        st.image(img_pil, use_container_width=True)
        
        # Tombol Pemicu
        if st.button("▶️ Run Analysis", type="primary"):
            st.session_state['run_analysis'] = True

    with col_results:
        st.subheader("Object and Caption")
        
        if st.session_state.get('run_analysis'):
            with st.spinner('Detect object...'):
                # 1. Deteksi YOLO
                detections = get_yolo_detections(img_bgr)
                # Filter kelas (Person, bicycle, car, motorcycle - COCO 1,2,3,18)
                allowed_classes = [0,1,2,16] # dr yolo (utk region proposal) 
                #SELECTED_CLASS = ["person", "bicycle", "car","dog"]
                label, bbox = predict_logic(img_bgr, svm_model, le, scaler, allowed_classes)
                # results = predict_multiple(img_bgr, svm_model, le, scaler, allowed_classes)

                if label and bbox:
                    # 3. Visualisasi
                    img_result = draw_bbox(img_pil, bbox, label)
                    # img_result = draw_multiple_bboxes(img_pil, results)
                    # st.image(img_result, caption=f"Detection result ({len(results)} objects)", use_container_width=True)
                    st.image(img_result, caption=f"Detection result: ", use_container_width=True)
                    
                    st.success(f"**Detected object:** `{label}`")
                    # all_labels = [obj['label'] for obj in results]
                    # label_str = ", ".join(set(all_labels))
                    # st.write(f"Detected Objects: `{label_str}`")

                    # 4. Generate Caption NLP
                    with st.spinner('Generate Caption...'):
                        llm_output = generate_llm_caption(label)
                        caption_text = llm_output.get("caption", "")
                        hashtags_text = llm_output.get("hashtags", "")

                        # st.info(f"**Caption:**\n\n_{caption}_")
                else:
                    st.warning("⚠️ object not found.")


            # --- BAGIAN OUTPUT CAPTION ---
            st.markdown("---")            
            with st.expander("Click to see the caption...", expanded=True):
                if caption_text != "" or hashtags_text != "":
                    # Tampilkan Caption dengan format Bold
                    st.markdown(f"{caption_text}")
                    # Tampilkan Hashtags dengan warna berbeda (info/code)
                    st.markdown(f"{hashtags_text}")
                else:
                    st.markdown("No caption generated, try again...")
            
            # Reset state agar tidak terus menerus berjalan
            st.session_state['run_analysis'] = False
        else:
            st.write("Press **Run Analysis**")
