# 🌾 AI Farmer Assistant — Crop Disease Detector

An AI-powered web app that detects crop diseases from leaf images and instantly suggests treatment, pesticides, and precautions — helping farmers make faster decisions.

**🔗 Live Demo:** [ai-crop-disease-detector.streamlit.app](https://ai-crop-disease-detector-xaqdebnnbd3q8qy48vcxjb.streamlit.app)

---

## ✨ Features

- 🔍 Instant disease detection from a leaf photo
- 🧪 Sample gallery to try the app without your own photo
- 📋 Treatment table — disease, pesticide, steps, and precautions
- 🔬 Confidence breakdown (top 3 predictions)
- 📥 Downloadable PDF diagnosis report
- 📱 Responsive, custom-styled UI

---

## 🧠 How It Works

Built using **Transfer Learning** with **MobileNetV2**, fine-tuned on the **PlantVillage dataset** (16,500+ images, 15 classes across Tomato, Potato, and Bell Pepper).

- **Validation Accuracy:** ~91%
- **Training:** 10 epochs on Google Colab (T4 GPU)

**Pipeline:** Upload leaf → resize & normalize → CNN prediction → treatment lookup → result + PDF report.

---

## 🛠️ Tech Stack

TensorFlow · Keras · MobileNetV2 · Streamlit · Pandas · fpdf2 · Custom CSS · Streamlit Community Cloud

---

## 📁 Folder Structure
crop-disease-detector/
├── app.py # Main Streamlit app
├── treatment_info.py # Disease → treatment/pesticide mapping
├── style.css # Custom UI styling
├── requirements.txt
├── runtime.txt
├── .streamlit/config.toml # Theme config
├── model/
│ ├── crop_disease_model.h5
│ └── class_names.json
└── samples/ # Demo images



---

## 🚀 Run Locally

```bash
git clone https://github.com/Yusuf8957/ai-crop-disease-detector.git
cd ai-crop-disease-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Limitations

- Trained on clean lab images — accuracy may drop on cluttered real-world backgrounds
- Single-label classifier — "Top 3 Matches" shows alternative guesses, not true multi-disease detection

---

## 👤 Author

**Yusuf** — built as a hands-on project covering the full ML pipeline, from training to deployment.

## 📄 License

MIT License — see [LICENSE](LICENSE)
