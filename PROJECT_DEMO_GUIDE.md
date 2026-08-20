# Crop Disease AI - Interview Demo Guide

## One-Line Pitch

Crop Disease AI is a farmer-friendly assistant that detects Tomato, Potato, and Bell Pepper leaf diseases, estimates infection severity, explains the AI decision with Grad-CAM, gives treatment guidance, and supports WhatsApp-based access through a free upload-link bridge.

## What Makes It Stand Out

- Crop-type verification before disease prediction to reduce false positives.
- Disease prediction with confidence score and top matches.
- Infection severity score using color segmentation.
- Treatment, pesticide, and precaution recommendations.
- Grad-CAM explainability so the result is not a black-box answer.
- Hindi voice summary for farmer accessibility.
- PDF diagnosis report for sharing or record keeping.
- Feedback collection for future offline retraining.
- WhatsApp bridge for a real-world farmer workflow.

## Local Web App Demo

```powershell
cd "C:\Users\moham\OneDrive\Desktop\AI Project\crop-disease-detector"
.\venv\Scripts\activate
streamlit run app.py
```

Demo flow:

1. Open the Streamlit URL.
2. Upload a Tomato, Potato, or Bell Pepper leaf image.
3. Show the disease name, confidence, severity bar, treatment, and precaution.
4. Open the Grad-CAM expander.
5. Download the PDF report.
6. Show the feedback section and explain future retraining.

## WhatsApp Bridge Demo

```powershell
cd "C:\Users\moham\OneDrive\Desktop\AI Project\crop-disease-detector"
.\venv\Scripts\activate
$env:PUBLIC_BASE_URL="https://YOUR-NGROK-URL.ngrok-free.app"
python whatsapp_bot.py
```

In another terminal:

```powershell
ngrok http 5000
```

Twilio Sandbox webhook:

```text
https://YOUR-NGROK-URL.ngrok-free.app/webhook
```

Demo flow:

1. Send `hi` to the Twilio WhatsApp Sandbox number.
2. Bot replies with the upload link.
3. Open the link, upload a leaf photo, and show the diagnosis report.
4. Explain that this avoids Twilio trial media restrictions while preserving the WhatsApp entry point.

## Architecture

```text
User
  -> Streamlit Web App (app.py)
  -> WhatsApp Bot / Upload Bridge (whatsapp_bot.py)
        -> Shared AI Diagnosis Engine (diagnosis_service.py)
              -> crop_type_classifier.h5
              -> crop_disease_model.h5
              -> treatment_info.py
```

## Interview Explanation

Use this answer:

> I built this as a practical farmer-assistance system, not just an image classifier. The app first verifies whether the uploaded image belongs to a supported crop, then predicts the disease, estimates severity, provides treatment guidance, explains the model decision with Grad-CAM, and generates a downloadable report. I also added a WhatsApp access bridge because farmers are more likely to use WhatsApp than a web dashboard. Since Twilio trial blocks media downloads, I implemented a free upload-link fallback that still uses the same AI diagnosis engine.

## Honest Limitations

- The model is trained mostly on clean PlantVillage-style images, so field images may be harder.
- Severity is estimated through color segmentation, not expert agronomy measurement.
- Feedback is collected for offline retraining; the deployed app does not retrain live.
- Twilio trial accounts cannot reliably download WhatsApp media, so the upload-link bridge is used for demos.

