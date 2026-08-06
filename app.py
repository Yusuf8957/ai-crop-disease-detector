import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import pandas as pd
import os
from treatment_info import treatment_info
from fpdf import FPDF
from datetime import datetime

# Model aur class names load karo
@st.cache_resource
def load_my_model():
    model = load_model('model/crop_disease_model.h5')
    with open('model/class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_my_model()

# Page config
st.set_page_config(page_title="AI Farmer Assistant", page_icon="🌾", layout="centered")

# CSS load
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# App UI
st.title("🌾 AI Farmer Assistant")
st.write("Upload a photo of your crop's leaf — AI will instantly detect the disease and suggest treatment.")

# ---------- About / How It Works Section ----------
with st.expander("ℹ️ How does this work?"):
    st.write("""
    This app uses **Transfer Learning** with **MobileNetV2** (pre-trained on ImageNet),
    fine-tuned on the **PlantVillage dataset** (16,500+ images across 15 classes covering
    Tomato, Potato, and Bell Pepper crops).

    - **Model architecture:** MobileNetV2 (frozen base) + custom classification layers
    - **Validation accuracy:** ~91%
    - **Training:** 10 epochs on Google Colab (T4 GPU)

    Upload a leaf photo below, or try one of the sample images to see it in action.
    """)

# ---------- Sample Images Gallery ----------
st.markdown("#### 🧪 Try a Sample")
sample_col1, sample_col2, sample_col3 = st.columns(3)

sample_paths = {
    "Sample 1": "samples/sample1.png",
    "Sample 2": "samples/sample2.png",
    "Sample 3": "samples/sample3.png",
}
selected_sample = None
with sample_col1:
    if os.path.exists(sample_paths["Sample 1"]):
        st.image(sample_paths["Sample 1"], use_container_width=True)
        if st.button("Use Sample 1"):
            selected_sample = sample_paths["Sample 1"]
with sample_col2:
    if os.path.exists(sample_paths["Sample 2"]):
        st.image(sample_paths["Sample 2"], use_container_width=True)
        if st.button("Use Sample 2"):
            selected_sample = sample_paths["Sample 2"]
with sample_col3:
    if os.path.exists(sample_paths["Sample 3"]):
        st.image(sample_paths["Sample 3"], use_container_width=True)
        if st.button("Use Sample 3"):
            selected_sample = sample_paths["Sample 3"]

st.markdown("#### 📷 Or Upload Your Own")
uploaded_file = st.file_uploader("Upload leaf photo", type=['jpg', 'jpeg', 'png'])

# Decide image source: uploaded file takes priority, else selected sample
img_source = None
if uploaded_file is not None:
    img_source = uploaded_file
elif selected_sample is not None:
    img_source = selected_sample

if img_source is not None:
    img = Image.open(img_source).convert('RGB')
    st.image(img, caption="Selected Leaf", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    with st.spinner("🔍 AI is analyzing..."):
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[str(predicted_index)]
        confidence = prediction[predicted_index] * 100

    info = treatment_info.get(predicted_class, {})
    disease_name = info.get("name", predicted_class)
    pesticide = info.get("pesticide", "Not available")
    treatment = info.get("treatment", "Treatment information not available.")
    precaution = info.get("precaution", "No specific precaution listed.")

    # ---------- Main Result ----------
    st.success(f"### 🔍 Diagnosis: {disease_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    if confidence < 60:
        st.warning("⚠️ Confidence is low — please try again with a clearer photo.")

    # ---------- Treatment Table ----------
    st.markdown("#### 📋 Recommendation Summary")
    table_data = {
        "Field": ["Detected Disease", "Recommended Pesticide", "Treatment Steps", "Precaution (Next Time)"],
        "Details": [disease_name, pesticide, treatment, precaution]
    }
    df = pd.DataFrame(table_data)
    st.table(df.set_index("Field"))

    # ---------- Top-3 Chart ----------
    st.markdown("#### 🔬 Confidence Breakdown (Top 3 Matches)")
    st.caption("The model checks against all known diseases — here are its top 3 guesses. This is not a true multi-disease detector, but shows what else the leaf could resemble.")

    top_3_indices = np.argsort(prediction)[-3:][::-1]
    chart_labels = []
    chart_values = []
    for idx in top_3_indices:
        cls = class_names[str(idx)]
        pct = prediction[idx] * 100
        name = treatment_info.get(cls, {}).get("name", cls)
        chart_labels.append(name)
        chart_values.append(round(pct, 1))

    chart_df = pd.DataFrame({"Confidence (%)": chart_values}, index=chart_labels)
    st.bar_chart(chart_df, color="#D4A24C")

    # ---------- Downloadable PDF Report ----------
    def generate_pdf_report():
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AI Farmer Assistant - Diagnosis Report", ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0,
            8,
            f"Generated on: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            ln=True,
            align="C",
        )

        pdf.ln(8)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, f"Detected Disease: {disease_name}", ln=True)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)

        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Recommended Pesticide:", ln=True)

        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, pesticide)

        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Treatment Steps:", ln=True)

        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, treatment)

        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Precaution (Next Time):", ln=True)

        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, precaution)

        return bytes(pdf.output())

    st.markdown("#### 📥 Download Report")

    pdf_bytes = generate_pdf_report()

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )