# 👟 Shoe Classifier - Classification de Chaussures avec Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-green?style=for-the-badge" />
</p>

<p align="center">
  <b>Application web de classification d'images de chaussures en 5 catégories</b>
</p>

---

## 🎯 Objectif

Classifier automatiquement des images de chaussures en **5 catégories** :

| Catégorie | Emoji |
|-----------|-------|
| Ballet Flat | 🩰 |
| Boat | ⛵ |
| Brogue | 👞 |
| Clog | 🥿 |
| Sneaker | 👟 |

## 🧠 Modèle

- **Architecture** : MobileNetV2 (Transfer Learning)
- **Fine-tuning** : 30 dernières couches débloquées
- **Dataset** : 13 000 images (10k train, 2.5k validation, 1.2k test)
- **Accuracy** : ~77% sur le jeu de test

## 🚀 Lancer l'application

### Prérequis

- Python 3.10+
- Le fichier modèle `shoes_mobilenetv2_finetuned.keras`

### Installation

```bash
# Cloner le repo
git clone https://github.com/Nassim-Bzr/Classification_CNN_DeepLearning.git
cd Classification_CNN_DeepLearning

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement

```bash
streamlit run app.py
```

Ouvrez votre navigateur à **http://localhost:8501**

## 📸 Captures d'écran

L'interface propose :
- Upload d'image par drag & drop
- Affichage des **Top-3 prédictions** avec barres de progression
- Thème sombre élégant

## 📁 Structure du projet

```
├── app.py                              # Application Streamlit
├── requirements.txt                    # Dépendances Python
├── Shoes classification_V2.ipynb       # Notebook d'entraînement
├── shoes_mobilenetv2_finetuned.keras   # Modèle entraîné (non inclus)
└── Shoes Dataset/                      # Dataset (non inclus)
    ├── Train/
    ├── Valid/
    └── Test/
```

## 🔧 Technologies utilisées

- **TensorFlow / Keras** - Entraînement du modèle
- **MobileNetV2** - Architecture de base (Transfer Learning)
- **Streamlit** - Interface web
- **Pillow** - Traitement d'images
- **NumPy** - Calculs numériques

## 📊 Résultats

| Modèle | Validation Accuracy | Test Accuracy |
|--------|---------------------|---------------|
| CNN from scratch | ~65% | ~63% |
| MobileNetV2 (gelé) | ~82% | - |
| MobileNetV2 (fine-tuned) | ~79% | **76.87%** |

## 👥 Auteurs

- Projet réalisé dans le cadre du TP Deep Learning - IPSSI

## 📄 Licence

Ce projet est à but éducatif.

---

<p align="center">
  Créé avec ❤️ en utilisant Streamlit & TensorFlow
</p>
