# 🌱 Crop Disease AI — AI Farmer Assistant

An AI-powered crop disease detection and farmer assistance platform that helps users identify crop diseases from leaf images and provides actionable treatment guidance.

The project combines **deep learning, explainable AI, severity analysis, farmer guidance, PDF reporting, feedback collection, WhatsApp integration, and an admin dashboard** into a single farmer-focused application.

---

## 🚀 Live Demo

### Streamlit Application
https://ai-crop-disease-detector-xaqdebnnbd3q8qy48vcxjb.streamlit.app/

### WhatsApp Upload Bridge
https://crop-disease-whatsapp.onrender.com

> The WhatsApp upload bridge is designed around the Twilio trial environment and provides a browser-based upload flow for sending leaf images to the AI diagnosis pipeline.

---

# 🎯 Project Goal

Farmers may not always have immediate access to agricultural experts or disease identification tools.

This project aims to provide a simple AI-assisted workflow:

```text
Farmer
   ↓
Upload Leaf Image
   ↓
Crop Verification
   ↓
Disease Classification
   ↓
Confidence Score
   ↓
Severity Analysis
   ↓
Explainable AI (Grad-CAM)
   ↓
Treatment Recommendation
   ↓
PDF Report / Feedback
```

The system is designed to make AI-based crop diagnosis easier to understand and use.

---

# ✨ Key Features

## 🔬 AI Crop Disease Detection

Upload a crop leaf image and the system predicts the most likely disease using a trained deep-learning model.

Supported crop/disease classes include:

- Pepper Bell Bacterial Spot
- Pepper Bell Healthy
- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Healthy

Total supported classes:

**15**

---

# 🌿 Crop Verification

The application contains a crop/leaf verification stage to help prevent irrelevant images from being processed as valid crop disease samples.

This improves the reliability of the diagnosis pipeline by separating image verification from disease classification.

---

# 🧠 Explainable AI — Grad-CAM

The application includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to make model predictions easier to understand.

Instead of only showing:

```text
Prediction: Tomato Early Blight
Confidence: 92%
```

the application can also visualize the regions of the leaf that received the most attention from the model.

Example:

```text
Original Leaf
       +
AI Attention / Grad-CAM
       ↓
Highlighted regions
```

This makes the model's prediction more interpretable during demonstrations and analysis.

---

# 📊 Confidence & Severity Analysis

The application provides additional information alongside the prediction, including:

- Prediction confidence
- Disease/severity information
- Treatment guidance
- Precautionary recommendations

The goal is to provide more useful information than a simple classification label.

---

# 💊 Treatment Guidance

After detecting a disease, the application provides treatment-oriented guidance based on the detected condition.

The system can provide:

- Treatment recommendation
- Pesticide requirement information
- Precautions
- Monitoring suggestions
- Preventive guidance

> The recommendations are intended as AI-assisted guidance and should not replace professional agricultural advice.

---

# 📄 PDF Report Generation

Users can generate a downloadable diagnosis report containing relevant prediction information.

The report can include:

- Uploaded image
- Crop/disease prediction
- Confidence
- Severity information
- Treatment recommendation
- Precautions

This makes the system more useful for documentation and sharing.

---

# 📱 WhatsApp AI Assistant

The project also includes a WhatsApp-assisted workflow.

### Workflow

```text
Farmer
   ↓
WhatsApp
   ↓
Upload / Receive Upload Link
   ↓
Leaf Image
   ↓
AI Diagnosis Engine
   ↓
Disease Result
   ↓
Treatment Guidance
```

The project uses a Flask-based upload bridge and integrates with the Twilio environment.

### Important

The current WhatsApp implementation uses a **free/trial-friendly upload-link bridge** because Twilio trial accounts have messaging restrictions.

The architecture separates the WhatsApp interface from the core AI diagnosis engine, making the diagnosis service reusable by other interfaces.

---

# 🔗 Shared AI Diagnosis Engine

The project contains a centralized diagnosis service:

```text
diagnosis_service.py
```

The shared service allows different interfaces to use the same AI logic.

For example:

```text
             ┌── Streamlit
             │
             ├── WhatsApp Bridge
             │
             └── Future API
                    │
                    ↓
          diagnosis_service.py
                    │
                    ↓
             AI Model Pipeline
```

This prevents duplicate prediction logic across different application interfaces.

---

# 🔐 Admin Authentication

The application contains a protected Admin section.

Admin functionality includes:

- Password-based authentication
- Feedback review
- Diagnosis feedback records
- Overall application ratings
- Model evaluation section
- Production architecture information

The normal user interface remains separate from the administrative interface.

---

# 📈 Model Evaluation Dashboard

The Admin dashboard includes a dedicated:

```text
Real Model Evaluation
```

section.

The evaluation pipeline is implemented using:

```text
evaluate_model.py
```

It supports generation of machine-readable evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

Evaluation metrics are intended to be generated from a **labeled test/evaluation dataset**.

The project does not treat historical training-curve or confusion-matrix images as live evaluation metrics.

---

# 🏗️ Production Architecture — 10,000+ Users

The project also includes a production scaling architecture describing how the application could be scaled for high traffic.

Current Streamlit deployment:

```text
User
  ↓
Streamlit
  ↓
AI Model
```

Production-oriented architecture:

```text
                 Users
            Web / WhatsApp
                   │
                   ▼
             CDN / WAF
                   │
                   ▼
          Load Balancer
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      API #1     API #2     API #N
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
            Redis / Job Queue
                   │
                   ▼
        ML Inference Workers
       Horizontal Scaling / GPU
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Object Storage      PostgreSQL /
     (Images)             Supabase
```

### Scaling principles

- Horizontal API scaling
- Load balancing
- Queue-based image processing
- Independent ML inference workers
- Object storage for images
- PostgreSQL/Supabase for application data
- Redis for asynchronous jobs
- Independent scaling of inference workers

### Important

The current Streamlit application is an **MVP/demo deployment**.

The architecture above is a **production scaling blueprint** and does not mean that the current localhost/Streamlit deployment has been load-tested with 10,000 concurrent users.

---

# 🧩 Application Architecture

```text
crop-disease-detector/
│
├── app.py
├── diagnosis_service.py
├── evaluate_model.py
├── whatsapp_bot.py
├── treatment_info.py
│
├── model/
│   ├── crop_disease_model.h5
│   ├── crop_type_classifier.h5
│   ├── leaf_classifier.h5
│   └── class_names.json
│
├── model_performance/
│
├── samples/
│   ├── sample1.png
│   └── sample2.png
│
├── .streamlit/
│   └── secrets configuration
│
├── style.css
├── PROJECT_DEMO_GUIDE.md
├── SCALING_ARCHITECTURE.md
├── LICENSE
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

## Frontend / Application

- Streamlit
- HTML/CSS
- Custom Streamlit styling

## Machine Learning

- Python
- TensorFlow
- Keras
- MobileNetV2-based models
- NumPy
- Pillow
- OpenCV
- Grad-CAM

## Backend / Integration

- Flask
- Twilio
- Requests

## Data / Storage

- Supabase
- PostgreSQL
- Pandas

## Reporting

- FPDF2

## Voice

- gTTS

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Yusuf8957/crop-disease-detector.git
cd crop-disease-detector
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv311
```

Activate:

```powershell
.\venv311\Scripts\activate
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

Activate the environment:

```powershell
.\venv311\Scripts\activate
```

Run:

```powershell
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 📱 Run the WhatsApp Upload Bridge

Open another terminal.

```powershell
.\venv\Scripts\activate
```

Run:

```powershell
python whatsapp_bot.py
```

Then expose the Flask server using ngrok:

```powershell
ngrok http 5000
```

The generated public URL can be used as the upload bridge.

Example:

```text
https://your-ngrok-url.ngrok-free.dev/upload
```

---

# 🔐 Configuration

Sensitive credentials should **not** be committed to GitHub.

For Streamlit secrets, use:

```text
.streamlit/secrets.toml
```

Example structure:

```toml
SUPABASE_URL = "your_supabase_url"
SUPABASE_SECRET_KEY = "your_supabase_secret_key"
```

Never commit real secret values.

Use:

```text
secrets.example.toml
```

as a template.

---

# 📊 Model Evaluation

The evaluation script supports:

```powershell
python evaluate_model.py --help
```

Basic usage:

```powershell
python evaluate_model.py --data "PATH_TO_LABELED_TEST_DATASET"
```

Optional model:

```powershell
python evaluate_model.py `
    --data "PATH_TO_LABELED_TEST_DATASET" `
    --model "model/crop_disease_model.h5"
```

The evaluation process generates machine-readable metrics for the Admin dashboard.

Expected output:

```text
model_performance/
└── evaluation_metrics.json
```

> Do not use unlabeled sample images as a substitute for a real test dataset when reporting model accuracy.

---

# 🧪 Example Prediction Flow

```text
1. User uploads leaf image
          ↓
2. Image validation
          ↓
3. Crop verification
          ↓
4. Disease model inference
          ↓
5. Confidence calculation
          ↓
6. Severity analysis
          ↓
7. Grad-CAM visualization
          ↓
8. Treatment guidance
          ↓
9. Optional PDF report
          ↓
10. Feedback collection
```

---

# 💡 Why This Project Is Different

This project is more than a simple image classifier.

A basic implementation would be:

```text
Image → Model → Disease
```

This project extends that workflow into:

```text
Image
  ↓
Crop Verification
  ↓
Disease Detection
  ↓
Confidence
  ↓
Severity
  ↓
Explainability
  ↓
Treatment Guidance
  ↓
PDF Report
  ↓
Feedback
```

It also provides multiple access/integration layers:

```text
                    ┌── Streamlit Web App
                    │
Farmer ─────────────┼── WhatsApp Bridge
                    │
                    └── Future API
                             │
                             ▼
                    Shared AI Engine
```

---

# 🎓 Interview Highlights

During an interview/demo, the project can demonstrate experience with:

### Machine Learning

- Image classification
- Transfer learning
- TensorFlow/Keras
- Model inference
- Confidence scoring
- Grad-CAM

### Software Engineering

- Modular architecture
- Shared AI service
- Backend integration
- API/upload bridge
- Authentication
- Error handling

### Production Thinking

- Horizontal scaling
- Load balancing
- Redis/job queues
- Independent inference workers
- Object storage
- PostgreSQL/Supabase

### Product Thinking

- Farmer-focused workflow
- Explainable predictions
- Treatment guidance
- PDF reports
- Feedback system
- Multiple access channels

---

# 🔒 Security Notes

Do not commit:

```text
.streamlit/secrets.toml
.env
API keys
Supabase secret keys
Twilio credentials
private credentials
```

Use:

```text
secrets.example.toml
```

for sharing configuration structure.

---

# ⚠️ Disclaimer

This application is an **AI-assisted crop disease detection system**.

Predictions and treatment suggestions should be treated as decision-support information and should not replace professional agricultural diagnosis or expert advice.

Image quality, lighting, crop variety, disease stage, and dataset limitations can affect prediction performance.

---

# 📌 Current Project Status

| Component | Status |
|---|---|
| AI Disease Classification | ✅ |
| Crop Verification | ✅ |
| Confidence Scoring | ✅ |
| Severity Analysis | ✅ |
| Grad-CAM Explainability | ✅ |
| Treatment Guidance | ✅ |
| PDF Reports | ✅ |
| Feedback Collection | ✅ |
| WhatsApp Upload Bridge | ✅ |
| Shared AI Diagnosis Service | ✅ |
| Admin Authentication | ✅ |
| Model Evaluation Pipeline | ✅ |
| Evaluation Dashboard | ✅ |
| 10,000+ User Architecture | ✅ Design |
| Actual 10,000-user Load Test | ⏳ Not performed |

---

# 📄 Documentation

Additional project documentation:

```text
PROJECT_DEMO_GUIDE.md
```

contains the interview/demo walkthrough.

```text
SCALING_ARCHITECTURE.md
```

contains the production scaling design.

---

# 👨‍💻 Author

**Mohammad Yusuf**

B.Tech CSE  
Integral University

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

## ⭐ Future Improvements

Potential future improvements include:

- Real production API deployment
- GPU-backed inference workers
- Redis-based asynchronous processing
- Automated model monitoring
- Real-time performance monitoring
- Larger and more diverse evaluation datasets
- Automated model retraining pipeline
- Better multilingual farmer support
- Production WhatsApp Business API integration
- Load testing with thousands of concurrent users
- Cloud-native deployment

---

## ⭐ If You Like This Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.
