import numpy as np
import pickle
from scipy.io import wavfile
from tensorflow.keras.models import load_model


# ==========================
# LOAD MODEL + ENCODER
# ==========================
model = load_model("models/panic_voice_model.h5")

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# ==========================
# FEATURE EXTRACTION (SAME AS TRAINING)
# ==========================
def extract_features(file_path):
    sr, audio = wavfile.read(file_path)

    audio = audio.astype(np.float32)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))

    max_len = 16000

    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    feature = audio.reshape(160, 100)
    feature = feature[..., np.newaxis]

    return feature


# ==========================
# PREDICT FUNCTION
# ==========================
def predict_audio(file_path):
    feature = extract_features(file_path)
    feature = feature.reshape(1, 160, 100, 1)

    prediction = model.predict(feature)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    emotion = encoder.inverse_transform([predicted_index])[0]

    return emotion, confidence


# ==========================
# TEST WITH YOUR AUDIO
# ==========================
if __name__ == "__main__":
    file_path = "test_audio/03-01-03-01-01-02-17.wav"  

    emotion, confidence = predict_audio(file_path)

    print("\n🎯 Prediction:", emotion)
    print("Confidence:", confidence)