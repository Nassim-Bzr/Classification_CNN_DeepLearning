"""
Application de Classification de Chaussures
Utilise MobileNetV2 pour classifier les images en 5 catégories
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ============================================================
# TÉLÉCHARGEMENT DU MODÈLE DEPUIS GOOGLE DRIVE
# ============================================================
MODEL_PATH = "shoes_mobilenetv2_finetuned.keras"
GDRIVE_FILE_ID = "18PI1vk2UtwJ13qdh3bgY7XmTwqc47qEo"

def download_model():
    """Télécharge le modèle depuis Google Drive si nécessaire"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle (~22 MB)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Shoe Classifier",
    page_icon="👟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS PERSONNALISÉ - DARK MODE ÉLÉGANT
# ============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Upload zone */
    .upload-zone {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .upload-zone:hover {
        border-color: #764ba2;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e, #252535);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .result-card.top-1 {
        border-left: 4px solid #00d4aa;
        background: linear-gradient(145deg, #1a2e2a, #252535);
    }

    .result-card.top-2 {
        border-left: 4px solid #667eea;
    }

    .result-card.top-3 {
        border-left: 4px solid #764ba2;
    }

    /* Progress bar */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
    }

    .progress-bar.top-1 {
        background: linear-gradient(90deg, #00d4aa, #00f5c4);
    }

    .progress-bar.top-2 {
        background: linear-gradient(90deg, #667eea, #7c8ff8);
    }

    .progress-bar.top-3 {
        background: linear-gradient(90deg, #764ba2, #9b6dcc);
    }

    /* Class name and percentage */
    .class-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
    }

    .percentage {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4aa;
        float: right;
    }

    /* Emoji icons */
    .shoe-emoji {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }

    /* Image container */
    .image-container {
        background: linear-gradient(145deg, #1e1e2e, #252535);
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTES
# ============================================================
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Ballet Flat', 'Boat', 'Brogue', 'Clog', 'Sneaker']
CLASS_EMOJIS = {
    'Ballet Flat': '🩰',
    'Boat': '⛵',
    'Brogue': '👞',
    'Clog': '🥿',
    'Sneaker': '👟'
}

# ============================================================
# CHARGEMENT DU MODÈLE (avec cache)
# ============================================================
@st.cache_resource
def load_model():
    """Charge le modèle MobileNetV2 pré-entraîné (télécharge si nécessaire)"""
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def predict_image(model, image):
    """
    Prédit la classe d'une image de chaussure.
    Retourne les top 3 prédictions avec leurs probabilités.
    """
    # Prétraitement de l'image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)

    # Conversion en RGB si nécessaire
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalisation et ajout de la dimension batch
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    predictions = model.predict(img_array, verbose=0)

    # Top 3 prédictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]

    results = []
    for idx in top3_indices:
        results.append({
            'class': CLASS_NAMES[idx],
            'emoji': CLASS_EMOJIS[CLASS_NAMES[idx]],
            'probability': float(predictions[0][idx])
        })

    return results

# ============================================================
# INTERFACE UTILISATEUR
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>👟 Shoe Classifier</h1>
    <p>Classification intelligente de chaussures avec Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Chargement du modèle
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle: {e}")
    model_loaded = False

if model_loaded:
    # Zone d'upload
    st.markdown("### 📤 Uploadez une image")
    uploaded_file = st.file_uploader(
        "Glissez-déposez une image ou cliquez pour sélectionner",
        type=['png', 'jpg', 'jpeg', 'webp'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🖼️ Image uploadée")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Résultats")

            # Prédiction
            with st.spinner("Analyse en cours..."):
                results = predict_image(model, image)

            # Affichage des résultats
            for i, result in enumerate(results):
                rank = i + 1
                prob_percent = result['probability'] * 100

                # Couleur selon le rang
                if rank == 1:
                    color_class = "top-1"
                    prob_color = "#00d4aa"
                elif rank == 2:
                    color_class = "top-2"
                    prob_color = "#667eea"
                else:
                    color_class = "top-3"
                    prob_color = "#764ba2"

                st.markdown(f"""
                <div class="result-card {color_class}">
                    <span class="percentage" style="color: {prob_color}">{prob_percent:.1f}%</span>
                    <p class="class-name">
                        <span class="shoe-emoji">{result['emoji']}</span>
                        {result['class']}
                    </p>
                    <div class="progress-container">
                        <div class="progress-bar {color_class}" style="width: {prob_percent}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Informations supplémentaires
        with st.expander("ℹ️ À propos de la prédiction"):
            st.write(f"""
            - **Modèle utilisé** : MobileNetV2 (fine-tuned)
            - **Classes disponibles** : {', '.join(CLASS_NAMES)}
            - **Confiance maximale** : {results[0]['probability']*100:.2f}%
            """)

    else:
        # Message d'attente
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.6);">
            <p style="font-size: 3rem; margin-bottom: 1rem;">👆</p>
            <p>Uploadez une image de chaussure pour commencer l'analyse</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Créé avec ❤️ en utilisant Streamlit & TensorFlow</p>
    <p>Modèle MobileNetV2 entraîné sur le dataset Shoes Classification</p>
</div>
""", unsafe_allow_html=True)
