import numpy as np
import pickle
from tensorflow.keras.models import load_model

# IMPORT YOUR FEATURE FUNCTION
from src.data_processing.feature_extraction import extract_features


# ==========================
# LOAD MODEL + ENCODER + SCALER
# ==========================
model = load_model("models/panic_voice_model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ==========================
# PREDICT FUNCTION
# ==========================
def predict_audio(file_path):
    feature = extract_features(file_path)

    if feature is None:
        return "Error", 0

    # Scale feature (IMPORTANT)
    feature = scaler.transform([feature])

    # Predict
    prediction = model.predict(feature)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    emotion = encoder.inverse_transform([predicted_index])[0]

    return emotion, confidence


# ==========================
# TEST
# ==========================
if __name__ == "__main__":
    file_path = "test_audio/03-01-03-01-01-02-17.wav"  

    emotion, confidence = predict_audio(file_path)

    print("\n🎯 Prediction:", emotion)
    print("Confidence:", confidence)