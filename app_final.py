import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Détecteur Déforestation", page_icon="🌳", layout="wide")

st.title("🌍 Détecteur de Déforestation - Burkina Faso")
st.markdown("Chargez une image satellite pour détecter les zones déforestées")

@st.cache_resource
def load_model():
    model_path = "modele_deforestation_final.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    return None

model = load_model()

with st.sidebar:
    st.header("📊 Performances du modèle")
    st.metric("Accuracy validation", "96.7%")
    st.metric("Précision test", "70.2%")
    st.markdown("---")
    st.markdown("**Entraîné sur :** Zone de Diébougou, Burkina Faso")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Image satellite", type=["jpg", "png", "jpeg", "tif"])

with col2:
    st.subheader("Détection")

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((64, 64))
    img_array = np.array(image) / 16.0
    img_input = img_array.reshape(1, 64, 64, 1)
    
    prediction = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(prediction[0])
    
    # Seuil de déforestation (classes > 8 = déforesté selon votre légende)
    if predicted_class > 8:
        st.error(f"🚨 ZONE DÉFORESTÉE (classe {predicted_class})")
    else:
        st.success(f"✅ ZONE BOISÉE (classe {predicted_class})")
    
    st.image(image, caption="Image analysée", width=200)
