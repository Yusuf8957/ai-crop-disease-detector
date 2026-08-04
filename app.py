import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
from treatment_info import treatment_info

# Model aur class names load karo (ek baar hi load ho, baar baar nahi)
@st.cache_resource
def load_my_model():
    model = load_model('model/crop_disease_model.h5')
    with open('model/class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_my_model()

# Page config
st.set_page_config(page_title="AI Farmer Assistant", page_icon="🌾", layout="centered")

# CSS file ko load karke apply karo
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# App UI
st.title("🌾 AI Farmer Assistant")
st.write("Upload a photo of your crop's leaf — AI will instantly detect the disease and suggest treatment.")

uploaded_file = st.file_uploader("📷 Upload leaf photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    with st.spinner("🔍 AI is analyzing..."):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[str(predicted_index)]
        confidence = np.max(prediction) * 100

    info = treatment_info.get(predicted_class, {})
    disease_name = info.get("name", predicted_class)
    treatment = info.get("treatment", "Treatment information not available.")

    st.success(f"### 🔍 Disease: {disease_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.info(f"**Treatment:** {treatment}")

    if confidence < 60:
        st.warning("⚠️ Confidence is low — please try again with a clearer photo.")