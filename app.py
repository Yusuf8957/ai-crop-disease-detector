import streamlit as st
import os
import json
import io
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from treatment_info import treatment_info
from fpdf import FPDF
from datetime import datetime

try:
    from supabase import create_client
except ImportError:
    create_client = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


# ==========================================================
# HELPER: safe markdown (fixes Streamlit's "indented text
# becomes a code block" bug when strings are indented in
# the Python source)
# ==========================================================

def md(text, unsafe=True):
    cleaned_lines = [line.strip() for line in text.strip("\n").splitlines()]
    cleaned = "\n".join(cleaned_lines)
    st.markdown(cleaned, unsafe_allow_html=unsafe)


def show_skeleton(container):
    container.markdown("""
    <div class="skeleton-card">
        <div class="skeleton skeleton-image"></div>
        <div class="skeleton skeleton-circle"></div>
        <div class="skeleton skeleton-line long"></div>
        <div class="skeleton skeleton-line medium"></div>
        <div class="skeleton skeleton-line short"></div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# SUPABASE — FEEDBACK COLLECTION ONLY
#
# Note: this deliberately does NOT include live model
# retraining on the deployed app. Streamlit Cloud's free
# tier isn't built for training workloads, and retraining
# on unreviewed public feedback risks silently degrading
# the model. Feedback (image + corrected label) is simply
# stored here so it can be reviewed and used to fine-tune
# the model offline (e.g. in Colab) later.
# ==========================================================

SUPABASE_BUCKET = "crop-ai-data"


@st.cache_resource
def get_supabase():
    if create_client is None:
        return None
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SECRET_KEY"])
    except Exception:
        return None


supabase = get_supabase()


def upload_feedback_image(image, original_filename):
    if supabase is None:
        return None
    safe_name = os.path.basename(original_filename).replace(" ", "_")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    image_path = (
        f"feedback/{stamp}_"
        f"{hashlib.sha256(safe_name.encode()).hexdigest()[:10]}_{safe_name}"
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=image_path,
            file=buffer.getvalue(),
            file_options={"content-type": "image/jpeg", "cache-control": "3600"},
        )
        return image_path
    except Exception:
        return None


def save_prediction_feedback(image, original_filename, predicted_class, corrected_class):
    if supabase is None:
        return False, "Feedback storage is not configured."

    image_path = upload_feedback_image(image, original_filename)
    if not image_path:
        return False, "Could not upload the image."

    row = {
        "image_path": image_path,
        "original_filename": original_filename,
        "predicted_class": predicted_class,
        "corrected_class": corrected_class,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        supabase.table("feedback").insert(row).execute()
        return True, image_path
    except Exception as e:
        return False, str(e)


def save_overall_feedback(rating, feedback_text):
    if supabase is None:
        return False, "Feedback storage is not configured."
    try:
        supabase.table("app_feedback").insert({
            "rating": rating,
            "feedback": feedback_text,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


def fetch_all_feedback():
    if supabase is None:
        return []
    try:
        return (
            supabase.table("feedback")
            .select("id,image_path,original_filename,predicted_class,corrected_class,created_at")
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception:
        return []


def fetch_all_app_feedback():
    if supabase is None:
        return []
    try:
        return (
            supabase.table("app_feedback")
            .select("id,rating,feedback,created_at")
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception:
        return []


def download_feedback_images_zip(rows):
    """Bundle every corrected-feedback image into a single ZIP the admin
    can download and use for offline fine-tuning (e.g. in Colab)."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest_lines = ["filename,corrected_class,predicted_class,created_at"]
        for row in rows:
            path = row.get("image_path")
            label = row.get("corrected_class") or "unlabeled"
            if not path:
                continue
            try:
                data = supabase.storage.from_(SUPABASE_BUCKET).download(path)
            except Exception:
                continue
            if not data:
                continue
            safe_label = str(label).replace("/", "_")
            filename = f"{safe_label}/{os.path.basename(path)}"
            zf.writestr(filename, data)
            manifest_lines.append(
                f"{filename},{row.get('corrected_class')},{row.get('predicted_class')},{row.get('created_at')}"
            )
        zf.writestr("manifest.csv", "\n".join(manifest_lines))
    buffer.seek(0)
    return buffer.getvalue()


def load_evaluation_metrics():
    """Load machine-readable evaluation results generated by evaluate_model.py.
    Returns None when no real evaluation artifact is present; the UI never
    invents accuracy numbers.
    """
    candidates = [
        Path("model_performance/evaluation_metrics.json"),
        Path("model_performance/metrics.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "accuracy" in data:
                    return data
            except Exception:
                pass
    return None


def render_evaluation_dashboard():
    """Render real evaluation metrics when a generated evaluation artifact exists."""
    st.markdown("### 🧪 Real Model Evaluation")
    metrics = load_evaluation_metrics()

    if metrics is None:
        st.warning(
            "No machine-readable test-set evaluation has been generated yet. "
            "Run `python evaluate_model.py --data <test_dataset>` to create "
            "model_performance/evaluation_metrics.json."
        )
        st.info(
            "The existing training-curves/confusion-matrix images are historical "
            "offline artifacts; this dashboard will not label them as live accuracy."
        )
        return

    st.caption(
        f"Evaluated on {metrics.get('dataset_samples', '—')} labeled images "
        f"from `{metrics.get('dataset_path', 'test dataset')}` on "
        f"{metrics.get('evaluated_at', '—')}."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Macro F1", f"{metrics.get('macro_f1', 0) * 100:.2f}%")
    c3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0) * 100:.2f}%")
    c4.metric("Test Images", str(metrics.get("dataset_samples", "—")))

    per_class = metrics.get("per_class") or {}
    if per_class:
        rows = []
        for label, values in per_class.items():
            rows.append({
                "Class": label,
                "Precision": round(values.get("precision", 0) * 100, 2),
                "Recall": round(values.get("recall", 0) * 100, 2),
                "F1": round(values.get("f1", 0) * 100, 2),
                "Support": values.get("support", 0),
            })
        st.markdown("#### Per-Class Metrics")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    report_path = Path("model_performance/classification_report.csv")
    if report_path.exists():
        st.download_button(
            "📄 Download Classification Report",
            data=report_path.read_bytes(),
            file_name="classification_report.csv",
            mime="text/csv",
        )


def render_scaling_architecture():
    st.markdown("### 🏗️ Production Architecture — 10,000+ Users")
    st.caption(
        "The current Streamlit + ngrok setup is a demo/MVP. This blueprint shows "
        "how the same AI engine should be deployed for high traffic."
    )
    st.code(
        """Users (Web / WhatsApp)\n
        "        │\n"
        "        ▼\n"
        " CDN / WAF / Load Balancer\n"
        "        │\n"
        "   ┌────┴─────┐\n"
        "   ▼          ▼\n"
        "API #1      API #2 ... API #N\n"
        "   │          │\n"
        "   └────┬─────┘\n"
        "        ▼\n"
        " Redis / Job Queue\n"
        "        │\n"
        "        ▼\n"
        " ML Inference Workers (horizontal scaling / GPU)\n"
        "        │\n"
        "   ┌────┴─────────┐\n"
        "   ▼              ▼\n"
        "Object Storage   PostgreSQL / Supabase\n"
        "(images)         (users, feedback, metrics)\n""",
        language="text",
    )
    cols = st.columns(4)
    cols[0].metric("API layer", "Horizontal")
    cols[1].metric("Inference", "Workers")
    cols[2].metric("Images", "Object Storage")
    cols[3].metric("State", "PostgreSQL")
    st.markdown(
        """
        **Scale-up plan:** containerize the inference API → add a load balancer →
        queue image jobs with Redis → run multiple model workers → store images in
        object storage → keep feedback/metadata in PostgreSQL/Supabase → add
        monitoring, rate limits, retries and autoscaling.
        """
    )


def render_admin_page():
    md('<div class="page-header"><span class="page-icon">🔐</span><h1 class="page-title">Admin — Feedback Review</h1></div>')

    password = st.text_input("Admin password", type="password", key="admin_password")
    if not password:
        st.info("Enter the admin password to view collected feedback.")
        return

    expected = st.secrets.get("ADMIN_PASSWORD", "")
    if not expected or password != expected:
        st.error("Incorrect password.")
        return

    if supabase is None:
        st.warning("Supabase is not configured, so there is nothing to review yet.")
        return

    st.session_state.admin_authenticated = True
    st.success("Access granted.")

    render_evaluation_dashboard()
    st.markdown("---")
    render_scaling_architecture()
    st.markdown("---")
    st.markdown("#### Diagnosis Feedback")
    rows = fetch_all_feedback()
    corrected_rows = [r for r in rows if r.get("corrected_class")]
    st.write(f"Total feedback entries: **{len(rows)}**  ·  Corrected (usable for retraining): **{len(corrected_rows)}**")

    if rows:
        table = pd.DataFrame([
            {
                "File": r.get("original_filename"),
                "Model said": r.get("predicted_class"),
                "Corrected to": r.get("corrected_class") or "(confirmed correct)",
                "Time": r.get("created_at"),
            }
            for r in rows
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)

        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download feedback as CSV",
            data=csv_bytes,
            file_name=f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        if corrected_rows:
            if st.button("📦 Prepare ZIP of corrected images (for offline retraining)"):
                with st.spinner("Packaging images..."):
                    zip_bytes = download_feedback_images_zip(corrected_rows)
                st.download_button(
                    "⬇️ Download ZIP",
                    data=zip_bytes,
                    file_name=f"corrected_feedback_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                )
                st.caption(
                    "Images are organized into folders by corrected label. "
                    "Use this in Colab to fine-tune the model offline, then "
                    "replace model/crop_disease_model.h5 and redeploy."
                )
    else:
        st.info("No feedback collected yet.")

    st.markdown("---")
    st.markdown("#### Overall App Ratings")
    app_rows = fetch_all_app_feedback()
    if app_rows:
        st.dataframe(pd.DataFrame(app_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No overall ratings submitted yet.")


# ==========================================================
# AI INSIGHTS — GRAD-CAM EXPLAINABILITY, SEVERITY SCORE,
# VOICE OUTPUT (HINDI)
#
# All three are defensive: if the model architecture or an
# optional dependency (opencv / gTTS) isn't available, these
# simply disable themselves instead of crashing the app.
# ==========================================================

CONV_LAYER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.DepthwiseConv2D,
)


def find_last_conv_layer(m):
    """Recursively find the name of the last convolutional layer, looking
    inside nested sub-models too — MobileNetV2 is often wrapped as a
    single nested layer when used as a feature extractor.

    We check the layer TYPE (Conv2D / DepthwiseConv2D / SeparableConv2D)
    rather than inspecting `.output_shape`, since that property can be
    unreliable (or raise) for layers with multiple inbound nodes — exactly
    the situation a reused nested base model creates — and its behavior
    also varies across Keras versions."""
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.Model):
            nested_name = find_last_conv_layer(layer)
            if nested_name:
                return nested_name
        if isinstance(layer, CONV_LAYER_TYPES):
            return layer.name
    return None


@st.cache_resource
def build_gradcam_model(_model):
    """Builds (and caches) a model mapping input images to
    (last conv layer activations, final predictions), used for Grad-CAM.
    Returns None if the architecture doesn't support it — callers handle
    that gracefully rather than crash the app."""
    last_conv_name = find_last_conv_layer(_model)
    if last_conv_name is None:
        return None

    conv_layer = None
    try:
        conv_layer = _model.get_layer(last_conv_name)
    except ValueError:
        for layer in _model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    conv_layer = layer.get_layer(last_conv_name)
                    break
                except ValueError:
                    continue

    if conv_layer is None:
        return None

    # Try the simplest approach first: `.output` works directly whenever
    # the layer has exactly one inbound node (which covers Keras 3, since
    # `get_output_at`/multi-node APIs from older Keras were removed there).
    # Only fall back to `get_output_at` for older Keras versions where a
    # reused nested layer genuinely has multiple inbound nodes.
    candidate_outputs = []
    try:
        candidate_outputs.append(conv_layer.output)
    except Exception:
        pass

    if hasattr(conv_layer, "get_output_at"):
        num_nodes = len(getattr(conv_layer, "_inbound_nodes", []))
        for node_index in reversed(range(max(num_nodes, 1))):
            try:
                candidate_outputs.append(conv_layer.get_output_at(node_index))
            except Exception:
                continue

    for conv_output in candidate_outputs:
        try:
            return tf.keras.models.Model(_model.inputs, [conv_output, _model.output])
        except Exception:
            continue

    return None


def make_gradcam_heatmap(img_array, grad_model, pred_index):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_pil, heatmap, alpha=0.45):
    base = np.array(img_pil.resize((224, 224)).convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap_color, alpha, base, 1 - alpha, 0)
    return Image.fromarray(overlay)


def calculate_severity(img_pil):
    """Estimates % of leaf area affected by disease using HSV color
    segmentation: green = healthy tissue, brown/yellow = diseased tissue."""
    img = np.array(img_pil.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([5, 30, 20])
    upper_brown = np.array([30, 255, 220])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    leaf_mask = cv2.bitwise_or(green_mask, brown_mask)
    leaf_area = int(np.count_nonzero(leaf_mask))
    diseased_area = int(np.count_nonzero(brown_mask))

    if leaf_area == 0:
        return 0.0

    severity_pct = (diseased_area / leaf_area) * 100
    return round(min(severity_pct, 100.0), 1)


def generate_voice_report(disease_name, confidence, severity_text, pesticide, lang="hi"):
    """Generates a spoken summary of the diagnosis using gTTS, in Hindi or English."""
    if lang == "hi":
        text = (
            f"निदान: {disease_name}. "
            f"विश्वास स्तर: {confidence:.0f} प्रतिशत। "
            f"संक्रमण की गंभीरता: {severity_text}। "
            f"अनुशंसित कीटनाशक: {pesticide}।"
        )
    else:
        text = (
            f"Diagnosis: {disease_name}. "
            f"Confidence level: {confidence:.0f} percent. "
            f"Infection severity: {severity_text}. "
            f"Recommended pesticide: {pesticide}."
        )
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.getvalue()


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
# MODEL LOADING (bundled files — no remote model syncing)
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

# Grad-CAM model is built once and cached; None if the architecture
# doesn't support it or opencv isn't installed — the UI degrades
# gracefully in that case.
gradcam_model = build_gradcam_model(model) if cv2 is not None else None

crop_type_labels = {0: "other", 1: "pepper", 2: "potato", 3: "tomato"}


# ==========================================================
# TOP NAVIGATION
# ==========================================================

# ==========================================================
# TOP NAVIGATION
# ==========================================================

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
    ("admin", "🔐 Admin"),
]

logo_col, brand_col, nav_col = st.columns(
    [0.55, 1.8, 6.5],
    vertical_alignment="center"
)

with logo_col:
    md('<div class="brand-logo">🌿</div>')

with brand_col:
    md(
        '<div class="brand-name">'
        '<span>Crop</span><span>Disease AI</span>'
        '</div>'
    )

with nav_col:
    nav_cols = st.columns(len(nav_items), gap="small")

    for col, (page_key, label) in zip(nav_cols, nav_items):
        with col:

            if st.button(
                label,
                key=f"nav_{page_key}",
                use_container_width=True
            ):

                # -----------------------------
                # ADMIN
                # -----------------------------
                if page_key == "admin":
                    st.query_params["admin"] = "1"
                    st.rerun()

                # -----------------------------
                # NORMAL PAGES
                # -----------------------------
                else:
                    # Remove admin mode before opening
                    # any normal navigation page.
                    st.query_params.clear()

                    st.session_state.page = page_key
                    st.rerun()

st.markdown(
    '<hr class="nav-divider">',
    unsafe_allow_html=True
)


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
            disease, along with a confidence score, an infection severity estimate,
            a Grad-CAM explanation of what the AI focused on, recommended pesticide,
            and preventive precautions — with a Hindi voice summary you can listen to.</p>
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
            <h4>Explainable AI (Grad-CAM)</h4>
            <p>A Grad-CAM heatmap overlay shows exactly which part of the leaf
            the model focused on to reach its diagnosis, instead of a
            black-box prediction.</p>
            <h4>Disease Severity Scoring</h4>
            <p>Color-segmentation analysis estimates what percentage of the
            leaf area is affected, turning a plain label into an actionable
            "spray now vs. monitor" signal.</p>
            <h4>Hindi Voice Diagnosis</h4>
            <p>Every result can be read aloud in Hindi, making the app more
            accessible to low-literacy users in the field.</p>
            <h4>Plant Health Report Generator</h4>
            <p>Generates a downloadable PDF report summarizing the diagnosis,
            recommended pesticide, treatment steps, and precautions for
            each uploaded leaf.</p>
            <h4>WhatsApp Access Bridge</h4>
            <p>A Flask + Twilio webhook lets farmers start from WhatsApp.
            If a trial account blocks direct media download, the bot sends
            a free upload link that still runs the same AI diagnosis engine.</p>
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
                <li>🔥 <strong>Explainable AI</strong> — how Grad-CAM highlights the regions
                the model used to make its decision</li>
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
            <p>🔥 Added Grad-CAM explainability — see exactly where the AI
            looked to make its diagnosis.</p>
            <p>🩺 Added a Disease Severity Score estimating % of leaf area
            affected, with clear spray/monitor guidance.</p>
            <p>🔊 Added Hindi voice output so results can be listened to,
            not just read.</p>
            <p>🌱 Added a crop-type verification step to reduce false positives
            on unsupported plant species.</p>
            <p>🤖 Improved diagnosis reporting with a downloadable PDF summary.</p>
            <p>📈 Added a Model Performance tab showing training curves and a
            confusion matrix.</p>
            <p>🚀 Multi-image batch upload — diagnose several leaves in one go.</p>
            <p>💬 Added feedback collection so incorrect diagnoses can be reviewed
            and used to improve the model in a future training pass.</p>
            <p>Added a WhatsApp upload-link bridge for farmer-friendly access
            without requiring paid media download during demos.</p>
        """,
    },
    "network": {
        "icon": "🌍",
        "title": "Contact & Network",
        "body": None,
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

# Admin page is accessed via a direct URL (?admin=1) and is
# intentionally NOT in the main navigation, so regular users
# never see it. It's password-protected and only supports
# reviewing/exporting feedback -- no live retraining here.
if st.query_params.get("admin") == "1":
    render_admin_page()
    st.stop()

if st.session_state.page != "home":
    render_subpage(st.session_state.page)
    st.stop()


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title("🌾 AI Farmer Assistant")
st.write("Upload photos of your crop's leaves — AI will instantly detect diseases and suggest treatment.")


# ==========================================================
# WHATSAPP AI ASSISTANT
# ==========================================================

st.markdown("---")

wa_col1, wa_col2 = st.columns([2.5, 1])

with wa_col1:
    st.markdown("### 📱 WhatsApp AI Assistant")
    st.write(
        "Get crop disease diagnosis through WhatsApp. "
        "Send a message to our bot and upload your leaf photo."
    )

    st.markdown("""
    **How it works:**

    💬 Send a message on WhatsApp  
    🔗 Receive the image upload link  
    📸 Upload your crop leaf  
    🤖 Get AI-powered disease diagnosis
    """)

st.link_button(
    "📸 Upload Leaf",
    "https://wa.me/17372212163?text=Hi"
)

st.caption(
        "WhatsApp users can send 'Hi' to the bot "
        "and receive the upload link."
    )

st.markdown("---")

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
            - **Explainability:** A Grad-CAM heatmap shows which part of the
              leaf the model focused on to reach its decision.
            - **Severity scoring:** Color segmentation estimates what % of the
              leaf area is affected, not just the disease name.
            - **Voice output:** Every diagnosis can be read aloud in Hindi.
            - **Batch support:** Upload multiple leaf photos at once to
              diagnose them together.
        """)

    st.markdown("#### 🧪 Try a Sample")

    sample_files = sorted(
        [
            path for path in [
                "samples/sample1.png",
                "samples/sample2.png",
                "samples/sample3.png",
            ]
            if os.path.exists(path)
        ]
    )

    # Label each sample using the number in its actual filename (e.g.
    # "samples/sample2.png" -> "Sample 2"), instead of re-numbering the
    # filtered list from 1 -- otherwise, if sample1.png is missing,
    # sample2.png would incorrectly get labeled "Sample 1".
    available_samples = {}
    for path in sample_files:
        filename = os.path.basename(path)
        digits = "".join(ch for ch in filename if ch.isdigit())
        label = f"Sample {digits}" if digits else filename
        available_samples[label] = path

    if available_samples:
        sample_cols = st.columns(len(available_samples))
        for col, (label, path) in zip(sample_cols, available_samples.items()):
            with col:
                st.image(path, use_container_width=True)
                if st.button(f"Use {label}", key=f"use_{label}"):
                    st.session_state.selected_sample = path
                    st.rerun()

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

        with col_result:
            loading_placeholder = col_result.empty()
            show_skeleton(loading_placeholder)

            # Threshold-based crop-type verification instead of plain argmax.
            # crop_type_labels: {0: "other", 1: "pepper", 2: "potato", 3: "tomato"}
            # Plain argmax was rejecting real Tomato/Potato/Pepper leaves whenever
            # "other" scored only marginally higher than the correct crop. We now
            # only reject as "other" if the model is genuinely confident (>60%);
            # otherwise we pick the best-scoring SUPPORTED crop.
            crop_prediction = crop_type_classifier.predict(img_array, verbose=0)[0]
            other_confidence = crop_prediction[0]
            if other_confidence > 0.60:
                predicted_crop = "other"
            else:
                supported_index = np.argmax(crop_prediction[1:]) + 1
                predicted_crop = crop_type_labels.get(supported_index, "other")

            if predicted_crop == "other":
                loading_placeholder.empty()
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
                loading_placeholder.empty()

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

                # ---------------- Disease Severity Score ----------------
                severity_pct = calculate_severity(img) if cv2 is not None else None
                if severity_pct is not None:
                    if severity_pct < 25:
                        sev_color, sev_label, sev_advice = "#2D6A4F", "Mild", "Monitor the plant — no urgent action needed."
                    elif severity_pct < 60:
                        sev_color, sev_label, sev_advice = "#D4A24C", "Moderate", "Spray the recommended pesticide soon."
                    else:
                        sev_color, sev_label, sev_advice = "#C0392B", "Severe", "Spray immediately — infection is spreading fast."

                    md(f"""
                    <div class="severity-card">
                        <div class="severity-header">
                            <span>🩺 Infection Severity</span>
                            <span class="severity-pct" style="color:{sev_color};">{severity_pct}%</span>
                        </div>
                        <div class="severity-bar-track">
                            <div class="severity-bar-fill" style="width:{severity_pct}%; background:{sev_color};"></div>
                        </div>
                        <div class="severity-footer">
                            <span class="severity-tag" style="background:{sev_color};">{sev_label}</span>
                            <span class="severity-advice">{sev_advice}</span>
                        </div>
                    </div>
                    """)

                # ---------------- Grad-CAM Explainability ----------------
                with st.expander("🔥 Why did the AI say this? (Grad-CAM)"):
                    if gradcam_model is not None:
                        try:
                            heatmap = make_gradcam_heatmap(img_array, gradcam_model, int(predicted_index))
                            if heatmap is not None:
                                overlay_img = overlay_gradcam(img, heatmap)
                                gc_col1, gc_col2 = st.columns(2)
                                with gc_col1:
                                    st.image(img.resize((224, 224)), caption="Original Leaf", use_container_width=True)
                                with gc_col2:
                                    st.image(overlay_img, caption="AI Attention (Grad-CAM)", use_container_width=True)
                                st.caption("Red/yellow regions are where the model focused most to reach this diagnosis.")
                            else:
                                st.caption("Grad-CAM visualization isn't available for this image.")
                        except Exception:
                            st.caption("Grad-CAM visualization isn't available for this model architecture.")
                    else:
                        st.caption(
                            "Grad-CAM isn't available — either opencv-python-headless isn't "
                            "installed, or this model's architecture doesn't expose a "
                            "compatible convolutional layer."
                        )

                # ---------------- Voice Output (Hindi / English) ----------------
                if gTTS is not None:
                    voice_state_key = f"voice_audio_{idx}_{img_label}"
                    lang_state_key = f"voice_lang_{idx}_{img_label}"

                    voice_lang_choice = st.radio(
                        "Voice language",
                        ["🇮🇳 Hindi", "🇬🇧 English"],
                        horizontal=True,
                        key=f"voice_lang_radio_{idx}_{img_label}",
                        label_visibility="collapsed",
                    )
                    selected_lang = "hi" if "Hindi" in voice_lang_choice else "en"

                    # If the language toggle changed since the last generated
                    # clip, drop the stale audio so the button regenerates it.
                    if st.session_state.get(lang_state_key) != selected_lang:
                        st.session_state[voice_state_key] = None
                        st.session_state[lang_state_key] = selected_lang

                    btn_label = "🔊 Diagnosis सुनें" if selected_lang == "hi" else "🔊 Listen to Diagnosis"
                    if st.button(btn_label, key=f"voice_btn_{idx}_{img_label}"):
                        with st.spinner("Generating audio..."):
                            try:
                                if selected_lang == "hi":
                                    severity_text = f"{severity_pct} प्रतिशत" if severity_pct is not None else "उपलब्ध नहीं"
                                else:
                                    severity_text = f"{severity_pct} percent" if severity_pct is not None else "not available"
                                audio_bytes = generate_voice_report(
                                    disease_name, confidence, severity_text, pesticide, lang=selected_lang
                                )
                                st.session_state[voice_state_key] = audio_bytes
                            except Exception:
                                st.session_state[voice_state_key] = None
                                error_msg = (
                                    "⚠️ आवाज़ अभी उपलब्ध नहीं है — internet connection जांचें।"
                                    if selected_lang == "hi"
                                    else "⚠️ Audio isn't available right now — check your internet connection."
                                )
                                st.caption(error_msg)
                    if st.session_state.get(voice_state_key):
                        st.audio(st.session_state[voice_state_key], format="audio/mp3")

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
                        if severity_pct is not None:
                            pdf.cell(0, 8, f"Infection Severity: {severity_pct}%", ln=True)
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

                # Feedback (collected for future offline retraining — no
                # live retraining happens in this app)
                feedback_key = f"fb_{idx}_{img_label}"
                if feedback_key not in st.session_state.feedback_given:
                    st.write("**Was this diagnosis correct?**")
                    fb_col1, fb_col2 = st.columns(2)
                    with fb_col1:
                        if st.button("👍 Yes", key=f"yes_{feedback_key}"):
                            ok, message = save_prediction_feedback(
                                img, img_label, predicted_class, predicted_class
                            )
                            st.session_state.feedback_given[feedback_key] = "yes"
                            if not ok and supabase is not None:
                                st.caption(f"(Feedback not saved: {message})")
                            st.rerun()
                    with fb_col2:
                        if st.button("👎 No", key=f"no_{feedback_key}"):
                            st.session_state.feedback_given[feedback_key] = "no"
                            st.rerun()
                elif st.session_state.feedback_given[feedback_key] == "no":
                    st.warning("What's the correct disease? This helps improve the model later.")
                    disease_options = [
                        (class_names[str(i)], treatment_info.get(class_names[str(i)], {}).get("name", class_names[str(i)]))
                        for i in range(len(class_names))
                    ]
                    option_labels = [name for _, name in disease_options]
                    selected_name = st.selectbox(
                        "Correct disease", option_labels, key=f"correct_{feedback_key}"
                    )
                    selected_class = next(k for k, name in disease_options if name == selected_name)
                    if st.button("💾 Save Correction", key=f"save_{feedback_key}"):
                        ok, message = save_prediction_feedback(
                            img, img_label, predicted_class, selected_class
                        )
                        st.session_state.feedback_given[feedback_key] = "saved"
                        if not ok and supabase is not None:
                            st.caption(f"(Correction not saved: {message})")
                        st.rerun()
                else:
                    st.caption("✅ Thanks for your feedback.")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 🕘 This Session's History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ==========================================================
# TAB 2 — MODEL PERFORMANCE
# ==========================================================

with tab_performance:

    st.markdown("### 📊 Model Evaluation & Performance")
    st.write(
        "Use the dashboard below for measured test-set metrics. "
        "The app does not fabricate accuracy values when an evaluation artifact is missing."
    )

    render_evaluation_dashboard()

    st.markdown("---")
    st.markdown("### 📈 Training Artifacts")

    perf_curve_path = "model_performance/training_curves.png"
    perf_matrix_path = "model_performance/confusion_matrix.png"

    if os.path.exists(perf_curve_path):
        st.markdown("#### Training Progress")
        st.image(perf_curve_path, use_container_width=True)
        st.caption("Historical training/validation curves from the offline training run.")
    else:
        st.info("Training curve image not found.")

    if os.path.exists(perf_matrix_path):
        st.markdown("#### Confusion Matrix")
        st.image(perf_matrix_path, use_container_width=True)
        st.caption("Historical confusion matrix from the offline evaluation run.")
    else:
        st.info("Confusion matrix image not found.")


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
        ok, message = save_overall_feedback(rating, feedback.strip())
        if ok:
            st.success(f"Thank you for your feedback! {rating}")
        else:
            st.success(f"Thank you for your feedback! {rating}")
            st.caption(f"(Not saved to database: {message})")
    else:
        st.warning("Please enter your feedback first.")