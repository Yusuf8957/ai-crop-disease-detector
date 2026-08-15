import streamlit as st
import os
import json
import textwrap
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.models import load_model
from treatment_info import treatment_info
from fpdf import FPDF
from datetime import datetime


# ==========================================================
# HELPER: safe markdown (fixes Streamlit's "indented text
# becomes a code block" bug when strings are indented in
# the Python source)
# ==========================================================

def md(text, unsafe=True):
    # Strip leading whitespace from EVERY line individually (not just the
    # common prefix). This is more robust than textwrap.dedent when nested
    # f-strings from different indentation levels get combined -- otherwise
    # Streamlit's markdown parser treats 4+ leading spaces as a code block,
    # which is what was causing parts of the HTML to render as raw text.
    cleaned_lines = [line.strip() for line in text.strip("\n").splitlines()]
    cleaned = "\n".join(cleaned_lines)
    st.markdown(cleaned, unsafe_allow_html=unsafe)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(page_title="AI Farmer Assistant", page_icon="🌾", layout="wide")


# ==========================================================
# LOAD CSS
# ==========================================================

def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "history": [],
    "feedback_given": {},
    "selected_sample": None,
    "processed_predictions": set(),
    "page": "home",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# MODEL LOADING
# ==========================================================

@st.cache_resource
def load_my_model():
    model = load_model("model/crop_disease_model.h5")
    with open("model/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_resource
def load_crop_type_classifier():
    return load_model("model/crop_type_classifier.h5")


model, class_names = load_my_model()
crop_type_classifier = load_crop_type_classifier()

crop_type_labels = {0: "other", 1: "pepper", 2: "potato", 3: "tomato"}


# ==========================================================
# TOP NAVIGATION
# ==========================================================

nav_items = [
    ("home", "🏠 Home"),
    ("about", "🌱 About"),
    ("projects", "🚀 Projects"),
    ("resources", "📚 Resources"),
    ("newsroom", "📰 Newsroom"),
    ("network", "🌍 Contact"),
]

logo_col, brand_col, nav_col = st.columns([0.55, 1.8, 6.5], vertical_alignment="center")

with logo_col:
    md('<div class="brand-logo">🌿</div>')

with brand_col:
    md('<div class="brand-name"><span>Crop</span><span>Disease AI</span></div>')

with nav_col:
    nav_cols = st.columns(len(nav_items), gap="small")
    for col, (page_key, label) in zip(nav_cols, nav_items):
        with col:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)


# ==========================================================
# SUB-PAGES (About / Projects / Resources / Newsroom / Network)
# ==========================================================

PAGE_CONTENT = {
    "about": {
        "icon": "🌱",
        "title": "About This Project",
        "body": """
            <p><strong>Crop Disease AI</strong> is an AI-powered application designed to detect crop
            diseases from leaf images.</p>
            <p>Upload a photo of a plant leaf and the model predicts the most likely
            disease, along with a confidence score, recommended pesticide, and
            preventive precautions.</p>
            <p>The application uses <strong>deep learning</strong> (Transfer Learning with
            MobileNetV2) and <strong>computer vision</strong> to help farmers and hobbyist
            growers quickly identify common crop diseases without needing to
            consult an expert in person.</p>
            <p>It currently supports <strong>Tomato, Potato, and Bell Pepper</strong> crops,
            covering 15 disease categories in total.</p>
        """,
    },
    "projects": {
        "icon": "🚀",
        "title": "Projects",
        "body": """
            <h4>Crop Disease Detection</h4>
            <p>AI-based crop disease classification using deep learning and
            computer vision. Achieves ~91% validation accuracy across 15 classes.</p>
            <h4>Crop-Type Verification</h4>
            <p>A secondary classifier checks whether an uploaded leaf actually
            belongs to a supported crop before running disease detection —
            reducing false, overconfident predictions on unrelated images.</p>
            <h4>Plant Health Report Generator</h4>
            <p>Generates a downloadable PDF report summarizing the diagnosis,
            recommended pesticide, treatment steps, and precautions for
            each uploaded leaf.</p>
        """,
    },
    "resources": {
        "icon": "📚",
        "title": "Resources",
        "body": """
            <ul>
                <li>🌿 <strong>Plant Disease Guide</strong> — general reference for common
                Tomato, Potato, and Pepper diseases</li>
                <li>🤖 <strong>AI &amp; Machine Learning</strong> — how Transfer Learning and CNNs
                work for image classification</li>
                <li>📊 <strong>Model Information</strong> — architecture, dataset, and training
                details (see the Model Performance tab)</li>
                <li>💡 <strong>Prevention Tips</strong> — crop rotation, spacing, watering habits,
                and fungicide/pesticide best practices</li>
            </ul>
        """,
    },
    "newsroom": {
        "icon": "📰",
        "title": "Newsroom",
        "body": """
            <p><strong>Latest Updates</strong></p>
            <p>🌱 Added a crop-type verification step to reduce false positives
            on unsupported plant species.</p>
            <p>🤖 Improved diagnosis reporting with a downloadable PDF summary.</p>
            <p>📈 Added a Model Performance tab showing training curves and a
            confusion matrix.</p>
            <p>🚀 Multi-image batch upload — diagnose several leaves in one go.</p>
        """,
    },
    "network": {
        "icon": "🌍",
        "title": "Contact & Network",
        "body": None,  # rendered specially below
    },
}

CONTACT_ITEMS = [
    ("📞", "Phone", "8957237058", "tel:+918957237058"),
    ("✉️", "Email", "m6645409@gmail.com", "mailto:m6645409@gmail.com"),
    ("💼", "LinkedIn", "linkedin.com/in/md-yusuf-iu", "https://www.linkedin.com/in/md-yusuf-iu"),
    ("🐙", "GitHub", "github.com/Yusuf8957", "https://github.com/Yusuf8957"),
]

PAGE_TAGS = {
    "about": ["AI FOR AGRICULTURE", "COMPUTER VISION"],
    "projects": ["MACHINE LEARNING", "DEPLOYMENT"],
    "resources": ["LEARNING", "REFERENCE"],
    "newsroom": ["UPDATES", "CHANGELOG"],
    "network": ["CONTACT", "COLLABORATE"],
}


def render_subpage(page_key):
    content = PAGE_CONTENT[page_key]
    tags = PAGE_TAGS.get(page_key, [])
    tags_html = "".join(f'<span class="page-tag">{t}</span>' for t in tags)

    breadcrumb = f'<div class="breadcrumb">Home &nbsp;›&nbsp; {content["title"]}</div>'

    hero = f"""
    <div class="hero-card">
        <div class="hero-left">
            {breadcrumb}
            <h1 class="page-title">{content['title']}</h1>
            <div class="tag-row">{tags_html}</div>
        </div>
        <div class="hero-right">
            <div class="hero-icon-panel">{content['icon']}</div>
        </div>
    </div>
    """
    md(hero)

    if page_key == "network":
        cards = ""
        for icon, label, value, link in CONTACT_ITEMS:
            cards += (
                f'<a class="contact-card" href="{link}" target="_blank" rel="noopener noreferrer">'
                f'<div class="contact-icon">{icon}</div>'
                f'<div class="contact-text">'
                f'<div class="contact-label">{label}</div>'
                f'<div class="contact-value">{value}</div>'
                f'</div></a>'
            )
        body = (
            '<div class="page-body">'
            '<p>Feel free to reach out for collaboration, questions, or feedback about this project.</p>'
            f'<div class="contact-grid">{cards}</div>'
            '</div>'
        )
    else:
        body_html = content["body"].strip()
        body = f'<div class="page-body">{body_html}</div>'

    md(body)

    if st.button("⬅ Back to App", key="back_home"):
        st.session_state.page = "home"
        st.rerun()


# ==========================================================
# ROUTER
# ==========================================================

if st.session_state.page != "home":
    render_subpage(st.session_state.page)
    st.stop()


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title("🌾 AI Farmer Assistant")
st.write("Upload photos of your crop's leaves — AI will instantly detect diseases and suggest treatment.")


# ==========================================================
# TABS
# ==========================================================

tab_diagnose, tab_performance = st.tabs(["🔍 Diagnose", "📊 Model Performance"])


# ==========================================================
# TAB 1 — DIAGNOSE
# ==========================================================

with tab_diagnose:

    with st.expander("ℹ️ How does this work?"):
        md("""
            This app uses **Transfer Learning** with **MobileNetV2** (pre-trained
            on ImageNet), fine-tuned on the **PlantVillage dataset** (16,500+
            images across 15 classes covering Tomato, Potato, and Bell Pepper
            crops).

            - **Model architecture:** MobileNetV2 (frozen base) + custom
              classification layers
            - **Validation accuracy:** ~91%
            - **Crop verification:** A separate classifier first checks whether
              each uploaded leaf belongs to Tomato, Potato, or Pepper — or is
              unsupported — before running disease detection.
            - **Batch support:** Upload multiple leaf photos at once to
              diagnose them together.
        """)

    # ------------------------------------------------------
    # SAMPLE GALLERY
    # ------------------------------------------------------

    st.markdown("#### 🧪 Try a Sample")

    sample_paths = {
        "Sample 1": "samples/sample1.png",
        "Sample 2": "samples/sample2.png",
        "Sample 3": "samples/sample3.png",
    }
    available_samples = {name: path for name, path in sample_paths.items() if os.path.exists(path)}

    if available_samples:
        sample_cols = st.columns(len(available_samples))
        for col, (label, path) in zip(sample_cols, available_samples.items()):
            with col:
                st.image(path, use_container_width=True)
                if st.button(f"Use {label}", key=f"use_{label}"):
                    st.session_state.selected_sample = path
                    st.rerun()

    # ------------------------------------------------------
    # UPLOAD IMAGE
    # ------------------------------------------------------

    st.markdown("#### 📷 Or Upload Your Own (multiple allowed)")

    uploaded_files = st.file_uploader(
        "Upload leaf photo(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    images_to_process = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            images_to_process.append((uploaded_file, uploaded_file.name))
    elif st.session_state.selected_sample:
        selected_path = st.session_state.selected_sample
        if os.path.exists(selected_path):
            images_to_process.append((selected_path, os.path.basename(selected_path)))

    # ------------------------------------------------------
    # PROCESS EACH IMAGE
    # ------------------------------------------------------

    for idx, (img_source, img_label) in enumerate(images_to_process):
        st.markdown("---")

        try:
            img = Image.open(img_source).convert("RGB")
        except Exception as e:
            st.error(f"Could not open {img_label}: {e}")
            continue

        col_img, col_result = st.columns([1, 1.4])
        with col_img:
            st.image(img, caption=img_label, use_container_width=True)

        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        with st.spinner(f"🔎 Verifying {img_label}..."):
            crop_prediction = crop_type_classifier.predict(img_array, verbose=0)[0]
        predicted_crop_index = np.argmax(crop_prediction)
        predicted_crop = crop_type_labels.get(predicted_crop_index, "other")

        with col_result:

            if predicted_crop == "other":
                st.error("🚫 This doesn't look like a supported crop leaf (Tomato, Potato, or Bell Pepper).")

                history_key = f"{img_label}_unsupported"
                if history_key not in st.session_state.processed_predictions:
                    st.session_state.history.append({
                        "Image": img_label,
                        "Result": "Unsupported / Not a crop leaf",
                        "Confidence": "-",
                        "Time": datetime.now().strftime("%I:%M %p"),
                    })
                    st.session_state.processed_predictions.add(history_key)

            else:
                prediction = model.predict(img_array, verbose=0)[0]
                predicted_index = np.argmax(prediction)
                predicted_class = class_names[str(predicted_index)]
                confidence = prediction[predicted_index] * 100

                info = treatment_info.get(predicted_class, {})
                disease_name = info.get("name", predicted_class)
                pesticide = info.get("pesticide", "Not available")
                treatment = info.get("treatment", "Treatment information not available.")
                precaution = info.get("precaution", "No specific precaution listed.")

                st.success(f"### 🔍 {disease_name}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                if confidence < 60:
                    st.warning("⚠️ Confidence is low — try a clearer photo.")

                history_key = f"{img_label}_{predicted_class}"
                if history_key not in st.session_state.processed_predictions:
                    st.session_state.history.append({
                        "Image": img_label,
                        "Result": disease_name,
                        "Confidence": f"{confidence:.1f}%",
                        "Time": datetime.now().strftime("%I:%M %p"),
                    })
                    st.session_state.processed_predictions.add(history_key)

                with st.expander("📋 Full Recommendation & Report"):
                    table_data = {
                        "Field": ["Detected Disease", "Recommended Pesticide", "Treatment Steps", "Precaution (Next Time)"],
                        "Details": [disease_name, pesticide, treatment, precaution],
                    }
                    recommendation_df = pd.DataFrame(table_data)
                    st.table(recommendation_df.set_index("Field"))

                    top_3_indices = np.argsort(prediction)[-3:][::-1]
                    chart_labels, chart_values = [], []
                    for i in top_3_indices:
                        cls = class_names[str(i)]
                        pct = prediction[i] * 100
                        name = treatment_info.get(cls, {}).get("name", cls)
                        chart_labels.append(name)
                        chart_values.append(round(pct, 1))
                    chart_df = pd.DataFrame({"Confidence (%)": chart_values}, index=chart_labels)
                    st.bar_chart(chart_df, color="#D4A24C")

                    def generate_pdf_report():
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Helvetica", "B", 16)
                        pdf.cell(0, 10, "AI Farmer Assistant - Diagnosis Report", ln=True, align="C")
                        pdf.set_font("Helvetica", "", 10)
                        pdf.cell(0, 8, "Generated on: " + datetime.now().strftime("%d %b %Y, %I:%M %p"), ln=True, align="C")
                        pdf.ln(8)
                        pdf.set_font("Helvetica", "B", 13)
                        pdf.cell(0, 10, f"Detected Disease: {disease_name}", ln=True)
                        pdf.set_font("Helvetica", "", 11)
                        pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
                        pdf.ln(4)
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.cell(0, 8, "Recommended Pesticide:", ln=True)
                        pdf.set_font("Helvetica", "", 11)
                        pdf.multi_cell(0, 7, str(pesticide))
                        pdf.ln(2)
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.cell(0, 8, "Treatment Steps:", ln=True)
                        pdf.set_font("Helvetica", "", 11)
                        pdf.multi_cell(0, 7, str(treatment))
                        pdf.ln(2)
                        pdf.set_font("Helvetica", "B", 12)
                        pdf.cell(0, 8, "Precaution (Next Time):", ln=True)
                        pdf.set_font("Helvetica", "", 11)
                        pdf.multi_cell(0, 7, str(precaution))
                        return bytes(pdf.output())

                    pdf_bytes = generate_pdf_report()
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"diagnosis_report_{idx}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}_{img_label}",
                    )

                feedback_key = f"fb_{idx}_{img_label}"
                if feedback_key not in st.session_state.feedback_given:
                    st.write("**Was this diagnosis correct?**")
                    fb_col1, fb_col2 = st.columns(2)
                    with fb_col1:
                        if st.button("👍 Yes", key=f"yes_{feedback_key}"):
                            st.session_state.feedback_given[feedback_key] = "yes"
                            st.rerun()
                    with fb_col2:
                        if st.button("👎 No", key=f"no_{feedback_key}"):
                            st.session_state.feedback_given[feedback_key] = "no"
                            st.rerun()
                else:
                    st.caption("✅ Thanks for your feedback: " + str(st.session_state.feedback_given[feedback_key]))

    # ------------------------------------------------------
    # PREDICTION HISTORY
    # ------------------------------------------------------

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 🕘 This Session's History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ==========================================================
# TAB 2 — MODEL PERFORMANCE
# ==========================================================

with tab_performance:

    st.markdown("### 📊 How Accurate Is This Model?")
    st.write("These charts show how the disease-detection model performed on the validation dataset (4,122 images it had never seen during training).")

    perf_curve_path = "model_performance/training_curves.png"
    perf_matrix_path = "model_performance/confusion_matrix.png"

    if os.path.exists(perf_curve_path):
        st.markdown("#### Training Progress")
        st.image(perf_curve_path, use_container_width=True)
        st.caption("Training vs. validation accuracy/loss across 10 epochs.")
    else:
        st.info("Training curve image not found.")

    if os.path.exists(perf_matrix_path):
        st.markdown("#### Confusion Matrix")
        st.image(perf_matrix_path, use_container_width=True)
        st.caption("Rows = actual disease, columns = predicted disease. Strong diagonal = accurate predictions.")
    else:
        st.info("Confusion matrix image not found.")

    st.markdown("#### Summary Metrics")
    summary_data = {
        "Metric": ["Overall Accuracy", "Macro Avg F1-Score", "Weighted Avg F1-Score", "Validation Samples"],
        "Value": ["90%", "0.88", "0.90", "4,122"],
    }
    st.table(pd.DataFrame(summary_data).set_index("Metric"))


# ==========================================================
# FOOTER
# ==========================================================

md("""
<div class="footer-bottom">
    <div class="footer-bottom-left">
        <div class="brand-logo footer-mini-logo">🌿</div>
        <div>
            <div class="footer-name">Crop Disease AI</div>
            <div class="footer-small">Made with care by Yusuf</div>
        </div>
    </div>
    <div class="footer-bottom-right">
        <a class="social-icon" href="tel:+918957237058" title="Phone">📞</a>
        <a class="social-icon" href="mailto:m6645409@gmail.com" title="Email">✉️</a>
        <a class="social-icon" href="https://www.linkedin.com/in/md-yusuf-iu" target="_blank" title="LinkedIn">💼</a>
        <a class="social-icon" href="https://github.com/Yusuf8957" target="_blank" title="GitHub">🐙</a>
    </div>
</div>
<div class="footer-copyright">© 2026 Crop Disease AI. Built as a learning project — not a substitute for professional agronomic advice.</div>
""")


# ==========================================================
# OVERALL STAR FEEDBACK
# ==========================================================

st.markdown("---")
st.markdown("## ⭐ Feedback")

rating = st.radio(
    "How would you rate your experience?",
    ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
    horizontal=True,
    key="overall_rating",
)

feedback = st.text_area("Your feedback", placeholder="Tell us what you think...", key="overall_feedback")

if st.button("Submit Feedback", key="submit_feedback"):
    if feedback.strip():
        st.success(f"Thank you for your feedback! {rating}")
    else:
        st.warning("Please enter your feedback first.")