# predict.py

import numpy as np
from tensorflow.keras.models import load_model
from utils.audio_processing import process_audio

# Load model once
model = load_model("model/panic_voice_model.h5")

def predict_panic(file_path):
    try:
        audio_data = process_audio(file_path)

        prediction = model.predict(audio_data)

        panic_score = float(prediction[0][0] * 100)

        emotion = "Panic" if panic_score > 60 else "Normal"

        return {
            "emotion": emotion,
            "panic_score": round(panic_score, 2)
        }

    except Exception as e:
        return {"error": str(e)}