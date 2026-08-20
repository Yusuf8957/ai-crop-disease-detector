"""
WhatsApp + web-upload bridge for Crop Disease AI.

This server supports two flows:
1. Official Twilio WhatsApp webhook for text replies.
2. Free trial-friendly upload page for leaf images when Twilio blocks media.

Run:
    python whatsapp_bot.py
    ngrok http 5000

Set the Twilio WhatsApp Sandbox incoming webhook to:
    https://YOUR-NGROK-URL.ngrok-free.app/webhook

Optional:
    set PUBLIC_BASE_URL=https://YOUR-NGROK-URL.ngrok-free.app
"""

import io
import os

import requests
from flask import Flask, render_template_string, request
from PIL import Image
from twilio.twiml.messaging_response import MessagingResponse

from diagnosis_service import diagnose_image, format_diagnosis_text


TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    print(
        "[WARNING] TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN are not set. "
        "Twilio media download will fail, but the upload-link fallback still works."
    )

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


UPLOAD_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Crop Disease AI Upload</title>
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: #f3f7f1;
            color: #142016;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        main {
            width: min(680px, 100%);
            background: #ffffff;
            border: 1px solid #d7e2d2;
            border-radius: 8px;
            padding: 22px;
            box-shadow: 0 10px 30px rgba(20, 32, 22, 0.08);
        }
        h1 {
            margin: 0 0 8px;
            font-size: 24px;
        }
        h2 {
            margin-top: 20px;
            font-size: 18px;
        }
        p {
            margin: 0 0 18px;
            line-height: 1.45;
        }
        input, button {
            width: 100%;
            box-sizing: border-box;
            font-size: 16px;
        }
        input {
            border: 1px solid #bfd0ba;
            border-radius: 6px;
            padding: 12px;
            background: #fbfdfb;
        }
        button {
            margin-top: 12px;
            border: 0;
            border-radius: 6px;
            padding: 12px 14px;
            background: #226b3a;
            color: white;
            font-weight: 700;
            cursor: pointer;
        }
        pre {
            white-space: pre-wrap;
            word-break: break-word;
            background: #f4f8f2;
            border: 1px solid #d7e2d2;
            border-radius: 6px;
            padding: 14px;
            line-height: 1.45;
        }
        .error {
            color: #9b1c1c;
        }
        .note {
            margin-top: 14px;
            color: #4a594c;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <main>
        <h1>Crop Disease AI</h1>
        <p>Tomato, Potato ya Bell Pepper leaf ki clear photo upload karein.</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Predict Disease</button>
        </form>
        <p class="note">WhatsApp trial media blocked ho to isi page se image upload karein.</p>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        {% if result %}
            <h2>Diagnosis Report</h2>
            <pre>{{ result }}</pre>
        {% endif %}
    </main>
</body>
</html>
"""


def get_public_upload_url() -> str:
    public_base_url = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if public_base_url:
        return f"{public_base_url}/upload"
    return request.url_root.rstrip("/") + "/upload"


def upload_instruction() -> str:
    upload_url = get_public_upload_url()
    return (
        "Namaste! Crop Disease AI ready hai.\n\n"
        "Twilio trial me WhatsApp photo download blocked ho sakta hai, "
        "isliye free upload link use karein:\n"
        f"{upload_url}\n\n"
        "Steps:\n"
        "1. Link open karo\n"
        "2. Leaf photo upload karo\n"
        "3. Browser me diagnosis report mil jayegi"
    )


def download_twilio_media(media_url: str) -> Image.Image:
    response = requests.get(
        media_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        timeout=20,
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


@app.route("/", methods=["GET"])
def home():
    return render_template_string(UPLOAD_PAGE)


@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if request.method == "GET":
        return render_template_string(UPLOAD_PAGE)

    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template_string(
            UPLOAD_PAGE,
            error="Please select a leaf image first.",
        )

    try:
        img = Image.open(uploaded_file.stream)
        result = diagnose_image(img)
        return render_template_string(
            UPLOAD_PAGE,
            result=format_diagnosis_text(result),
        )
    except Exception as e:
        return render_template_string(
            UPLOAD_PAGE,
            error=f"Image process nahi ho payi: {type(e).__name__}: {e}",
        )


@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    response = MessagingResponse()
    message = response.message()

    num_media = int(request.values.get("NumMedia", 0))
    if num_media == 0:
        message.body(upload_instruction())
        return str(response)

    media_url = request.values.get("MediaUrl0")
    try:
        img = download_twilio_media(media_url)
        result = diagnose_image(img)
        reply_text = format_diagnosis_text(result)
    except Exception as e:
        if (
            isinstance(e, requests.HTTPError)
            and e.response is not None
            and e.response.status_code == 401
        ):
            reply_text = upload_instruction()
        else:
            reply_text = (
                "Photo process nahi ho payi, dobara try karein.\n"
                f"({type(e).__name__}: {e})\n\n"
                f"Backup upload link: {get_public_upload_url()}"
            )

    message.body(reply_text)
    return str(response)


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
