import numpy as np
import pickle
from tensorflow.keras.models import load_model

from src.data_processing.feature_extraction import extract_features

# 🔹 Load model & tools
model = load_model("models/panic_voice_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


def predict_audio(file_path):
    features = extract_features(file_path)

    if features is None:
        print("❌ Feature extraction failed")
        return

    # 🔥 IMPORTANT: scale features
    features = scaler.transform([features])

    # 🔹 Predict
    preds = model.predict(features)[0]

    # 🔍 Debug
    print("Raw output:", preds)

    predicted_index = np.argmax(preds)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = preds[predicted_index]

    print(f"\n🎯 Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")


# 🔹 Test with your file
if __name__ == "__main__":
    test_file = "test_audio/YAF_youth_disgust.wav"   
    predict_audio(test_file)