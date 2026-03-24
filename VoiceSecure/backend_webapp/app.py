from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils.audio_processing import process_audio
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/panic_voice_model.h5")

@app.route("/")
def home():
    return "VoiceSecure Backend Running"

@app.route("/detect-panic", methods=["POST"])
def detect_panic():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        audio_data = process_audio(file_path)

        print("Audio shape:", audio_data.shape)

        prediction = model.predict(audio_data)

        print("Prediction:", prediction)

        # SAFE extraction
        panic_score = float(prediction[0][0] * 100)

        emotion = "Panic" if panic_score > 60 else "Normal"

        response = {
            "emotion": str(emotion),
            "panic_score": round(panic_score, 2)
        }

        print("Sending:", response)

        return jsonify(response)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)