"""
Application de Classification de Chaussures
Utilise MobileNetV2 pour classifier les images en 5 categories
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
from datetime import datetime, timedelta
import io
import base64
import json

# ============================================================
# GESTION DE L'HISTORIQUE PERSISTANT (72h)
# ============================================================
HISTORY_FILE = "prediction_history.json"
HISTORY_TTL_HOURS = 72

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
            now = datetime.now()
            valid_history = []
            for item in data:
                item_time = datetime.fromisoformat(item.get('timestamp', '2000-01-01'))
                if now - item_time < timedelta(hours=HISTORY_TTL_HOURS):
                    valid_history.append(item)
            return valid_history
    except Exception:
        pass
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception:
        pass

# ============================================================
# TELECHARGEMENT DU MODELE DEPUIS GOOGLE DRIVE
# ============================================================
MODEL_PATH = "shoes_mobilenetv2_finetuned.keras"
GDRIVE_FILE_ID = "18PI1vk2UtwJ13qdh3bgY7XmTwqc47qEo"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Telechargement du modele (~22 MB)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Shoe Classifier AI",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISE - DESIGN WOW
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background:
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 212, 170, 0.05) 0%, transparent 30%);
        pointer-events: none;
        z-index: -1;
    }

    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), 0 0 100px rgba(118, 75, 162, 0.2);
        position: relative;
        overflow: hidden;
        animation: headerGlow 3s ease-in-out infinite;
    }

    @keyframes headerGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), 0 0 100px rgba(118, 75, 162, 0.2); }
        50% { box-shadow: 0 20px 80px rgba(102, 126, 234, 0.6), 0 0 120px rgba(118, 75, 162, 0.4); }
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }

    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 300;
    }

    .stat-card {
        background: linear-gradient(145deg, rgba(30, 30, 46, 0.9), rgba(37, 37, 53, 0.9));
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .result-card {
        background: linear-gradient(145deg, rgba(30, 30, 46, 0.95), rgba(37, 37, 53, 0.95));
        border-radius: 20px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .result-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    }

    .result-card.top-1 {
        border-left: 5px solid #00d4aa;
        background: linear-gradient(145deg, rgba(0, 50, 40, 0.3), rgba(37, 37, 53, 0.95));
        animation: slideIn 0.5s ease-out, pulseGreen 2s infinite;
    }

    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 170, 0.2); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 170, 0.4); }
    }

    .result-card.top-2 { border-left: 5px solid #667eea; animation: slideIn 0.6s ease-out; }
    .result-card.top-3 { border-left: 5px solid #764ba2; animation: slideIn 0.7s ease-out; }

    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        height: 14px;
        overflow: hidden;
        margin-top: 0.8rem;
    }

    .progress-bar {
        height: 100%;
        border-radius: 15px;
        animation: progressFill 1s ease-out;
        position: relative;
        overflow: hidden;
    }

    @keyframes progressFill { from { width: 0; } }

    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: progressShine 2s infinite;
    }

    @keyframes progressShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .progress-bar.top-1 { background: linear-gradient(90deg, #00d4aa, #00f5c4, #00d4aa); background-size: 200% 100%; }
    .progress-bar.top-2 { background: linear-gradient(90deg, #667eea, #7c8ff8, #667eea); background-size: 200% 100%; }
    .progress-bar.top-3 { background: linear-gradient(90deg, #764ba2, #9b6dcc, #764ba2); background-size: 200% 100%; }

    .class-name { font-size: 1.4rem; font-weight: 600; color: #ffffff; margin: 0; }
    .percentage { font-size: 2rem; font-weight: 800; float: right; }
    .percentage.top-1 { color: #00d4aa; text-shadow: 0 0 20px rgba(0, 212, 170, 0.5); }
    .percentage.top-2 { color: #667eea; }
    .percentage.top-3 { color: #764ba2; }
    .shoe-emoji { font-size: 1.8rem; margin-right: 0.8rem; }

    .history-item {
        background: linear-gradient(145deg, rgba(30, 30, 46, 0.9), rgba(37, 37, 53, 0.9));
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .history-item:hover {
        transform: scale(1.05);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }

    .history-item img { width: 100px; height: 100px; border-radius: 12px; object-fit: cover; margin-bottom: 0.5rem; }
    .history-label { font-size: 1rem; font-weight: 600; color: #ffffff; }
    .history-confidence { font-size: 0.9rem; color: #00d4aa; font-weight: 600; }
    .history-time { font-size: 0.75rem; color: rgba(255,255,255,0.5); }

    .modal-container { text-align: center; padding: 1rem; }
    .modal-image {
        width: 100%;
        border-radius: 20px;
        margin: 0 auto 1.5rem auto;
        display: block;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        border: 3px solid rgba(255,255,255,0.1);
    }
    .modal-label { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin: 1rem 0; }
    .modal-confidence { font-size: 1.5rem; color: #00d4aa; font-weight: 600; margin-bottom: 0.5rem; }
    .modal-date { font-size: 1.1rem; color: rgba(255,255,255,0.6); margin-bottom: 1rem; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 15px;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), rgba(118, 75, 162, 0.5), transparent);
        margin: 2.5rem 0;
    }

    .footer { text-align: center; padding: 2rem; color: rgba(255,255,255,0.4); font-size: 0.9rem; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .success-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d4aa, #00f5c4);
        color: #000;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        animation: bounce 0.5s ease;
        margin-top: 1rem;
    }

    @keyframes bounce {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }

    .not-shoe-card {
        background: linear-gradient(145deg, rgba(255, 82, 82, 0.15), rgba(37, 37, 53, 0.95));
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 2px solid rgba(255, 82, 82, 0.5);
        text-align: center;
        animation: shakeAndFade 0.6s ease-out;
    }

    @keyframes shakeAndFade {
        0% { opacity: 0; transform: translateX(-10px); }
        20% { transform: translateX(10px); }
        40% { transform: translateX(-10px); }
        60% { transform: translateX(10px); }
        80% { transform: translateX(-5px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    .not-shoe-card:hover {
        box-shadow: 0 0 30px rgba(255, 82, 82, 0.3);
    }

    .not-shoe-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: wobble 2s infinite;
    }

    @keyframes wobble {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-10deg); }
        75% { transform: rotate(10deg); }
    }

    .not-shoe-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff5252;
        margin: 0.5rem 0;
        text-shadow: 0 0 20px rgba(255, 82, 82, 0.3);
    }

    .not-shoe-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        margin: 0.5rem 0;
    }

    .detected-class {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        color: #ffd54f;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTES
# ============================================================
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Ballet Flat', 'Boat', 'Brogue', 'Clog', 'Sneaker']
CLASS_EMOJIS = {'Ballet Flat': '🩰', 'Boat': '⛵', 'Brogue': '👞', 'Clog': '🥿', 'Sneaker': '👟'}
CLASS_FR = {
    'Ballet Flat': 'Ballerine',
    'Boat': 'Chaussure Bateau',
    'Brogue': 'Richelieu',
    'Clog': 'Sabot',
    'Sneaker': 'Basket'
}

# Classes ImageNet liées aux chaussures (indices dans ImageNet)
IMAGENET_SHOE_CLASSES = {
    502: 'clog',
    518: 'cowboy_boot',
    519: 'cowboy_hat',  # exclure
    539: 'dutch_oven',  # exclure
    542: 'Running_shoe',
    543: 'shoe_shop',
    614: 'loafer',
    630: 'Loafer',
    770: 'running_shoe',
    774: 'sandal',
    787: 'slipper',
    788: 'sneaker',
    795: 'boot',
}
# Indices réels des chaussures dans ImageNet
SHOE_INDICES = [502, 518, 542, 614, 770, 774, 787, 788, 795]
CONFIDENCE_THRESHOLD = 0.15  # Seuil minimum pour considérer que c'est une chaussure

# ============================================================
# POPUP POUR L'HISTORIQUE
# ============================================================
@st.dialog("Details de la prediction", width="large")
def show_history_detail(item):
    st.markdown(f"""
    <div class="modal-container">
        <img src="data:image/png;base64,{item['image_base64']}" class="modal-image"/>
        <p class="modal-label">{item['emoji']} {item['label']}</p>
        <p class="modal-confidence">Confiance : {item['confidence']*100:.1f}%</p>
        <p class="modal-date">Envoye le {item.get('date', 'N/A')} a {item.get('time', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Fermer", use_container_width=True):
            st.rerun()

# ============================================================
# CHARGEMENT DU MODELE
# ============================================================
@st.cache_resource
def load_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_imagenet_model():
    """Charge MobileNetV2 pré-entraîné sur ImageNet pour la détection"""
    return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# ============================================================
# VERIFICATION SI L'IMAGE EST UNE CHAUSSURE
# ============================================================
def is_shoe_image(imagenet_model, image):
    """
    Vérifie si l'image contient une chaussure en utilisant ImageNet.
    Retourne (is_shoe: bool, detected_class: str, confidence: float)
    """
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)

    # Gérer les images en niveaux de gris ou avec canal alpha
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Prétraitement pour MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction ImageNet
    predictions = imagenet_model.predict(img_array, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

    # Vérifier si une des top prédictions est une chaussure
    shoe_keywords = ['shoe', 'boot', 'sandal', 'loafer', 'clog', 'sneaker', 'slipper', 'footwear']

    for _, pred_name, pred_conf in decoded:
        pred_name_lower = pred_name.lower().replace('_', ' ')
        for keyword in shoe_keywords:
            if keyword in pred_name_lower:
                return True, pred_name.replace('_', ' '), float(pred_conf)

    # Retourner la classe la plus probable si ce n'est pas une chaussure
    top_pred = decoded[0]
    return False, top_pred[1].replace('_', ' '), float(top_pred[2])

# ============================================================
# FONCTION DE PREDICTION
# ============================================================
def predict_image(model, image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    return [{'class': CLASS_NAMES[idx], 'class_fr': CLASS_FR[CLASS_NAMES[idx]], 'emoji': CLASS_EMOJIS[CLASS_NAMES[idx]], 'probability': float(predictions[0][idx])} for idx in top3_indices]

# ============================================================
# SIDEBAR - STATISTIQUES
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Statistiques")
    if 'history' not in st.session_state:
        st.session_state.history = load_history()
    total = len(st.session_state.history)
    class_counts = {}
    total_conf = 0
    for item in st.session_state.history:
        class_counts[item.get('label', '')] = class_counts.get(item.get('label', ''), 0) + 1
        total_conf += item.get('confidence', 0)
    avg_conf = (total_conf / total * 100) if total > 0 else 0

    st.markdown(f'<div class="stat-card"><div class="stat-number">{total}</div><div class="stat-label">Predictions</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="stat-card"><div class="stat-number">{avg_conf:.1f}%</div><div class="stat-label">Confiance moy.</div></div>', unsafe_allow_html=True)

    if class_counts:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Par categorie")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"{CLASS_EMOJIS.get(cls, '👟')} **{cls}**: {count}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ A propos")
    st.markdown("**Modele**: MobileNetV2\n**Precision**: ~77%")

# ============================================================
# INTERFACE PRINCIPALE
# ============================================================
st.markdown('<div class="main-header"><h1>👟 Shoe Classifier AI</h1><p>Classification intelligente avec Deep Learning</p></div>', unsafe_allow_html=True)

try:
    model = load_model()
    imagenet_model = load_imagenet_model()
    model_loaded = True
except Exception as e:
    st.error(f"Erreur: {e}")
    model_loaded = False

if model_loaded:
    st.markdown("### 📤 Uploadez une image")
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### 🖼️ Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Resultats")
            with st.spinner("🔍 Analyse en cours..."):
                # Vérifier d'abord si c'est une chaussure
                is_shoe, detected_class, detected_conf = is_shoe_image(imagenet_model, image)

            if not is_shoe:
                # Ce n'est pas une chaussure !
                st.markdown(f'''
                <div class="not-shoe-card">
                    <div class="not-shoe-emoji">🤔</div>
                    <p class="not-shoe-title">Oups ! Ce n'est pas une chaussure</p>
                    <p class="not-shoe-subtitle">L'image ne semble pas contenir de chaussure.</p>
                    <div class="detected-class">Detecte : {detected_class} ({detected_conf*100:.1f}%)</div>
                </div>
                ''', unsafe_allow_html=True)
                st.info("💡 Essayez avec une photo de chaussure : sneaker, ballerine, mocassin, sabot...")
            else:
                # C'est une chaussure, on classifie
                results = predict_image(model, image)

                if results[0]['probability'] > 0.85:
                    st.balloons()
                    st.markdown('<div class="success-badge">🎉 Haute confiance!</div>', unsafe_allow_html=True)

                file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name
                if not any(h.get('file_id') == file_id for h in st.session_state.history):
                    img_thumb = image.copy()
                    img_thumb.thumbnail((200, 200))
                    buffered = io.BytesIO()
                    img_thumb.save(buffered, format="PNG")
                    now = datetime.now()
                    st.session_state.history.insert(0, {
                        'file_id': file_id,
                        'image_base64': base64.b64encode(buffered.getvalue()).decode(),
                        'label': results[0]['class'],
                        'emoji': results[0]['emoji'],
                        'confidence': results[0]['probability'],
                        'time': now.strftime("%H:%M:%S"),
                        'date': now.strftime("%d/%m/%Y"),
                        'timestamp': now.isoformat()
                    })
                    st.session_state.history = st.session_state.history[:10]
                    save_history(st.session_state.history)

                for i, r in enumerate(results):
                    c = ["top-1", "top-2", "top-3"][i]
                    st.markdown(f'''<div class="result-card {c}">
                        <span class="percentage {c}">{r["probability"]*100:.1f}%</span>
                        <p class="class-name"><span class="shoe-emoji">{r["emoji"]}</span>{r["class"]}</p>
                        <div class="progress-container"><div class="progress-bar {c}" style="width:{r["probability"]*100}%"></div></div>
                    </div>''', unsafe_allow_html=True)

    else:
        st.markdown('<div style="text-align:center;padding:4rem;color:rgba(255,255,255,0.6);"><p style="font-size:5rem;">👟</p><p style="font-size:1.3rem;">Glissez une image pour commencer</p></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📜 Historique")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🗑️ Effacer"):
                st.session_state.history = []
                save_history([])
                st.rerun()
        with c2:
            st.caption(f"💾 {len(st.session_state.history)} predictions (expire 72h)")

        cols = st.columns(5, gap="medium")
        for idx, item in enumerate(st.session_state.history):
            with cols[idx % 5]:
                st.markdown(f'<div class="history-item"><img src="data:image/png;base64,{item["image_base64"]}"/><div class="history-label">{item["emoji"]} {item["label"]}</div><div class="history-confidence">{item["confidence"]*100:.1f}%</div><div class="history-time">{item.get("date","")} {item.get("time","")}</div></div>', unsafe_allow_html=True)
                if st.button("👁️", key=f"v_{idx}", use_container_width=True):
                    show_history_detail(item)

st.markdown('<div class="footer"><p>Powered by TensorFlow & Streamlit</p></div>', unsafe_allow_html=True)
