import streamlit as st
import os
import sys

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.model.predict import predict_audio

# 🎨 PAGE CONFIG
st.set_page_config(page_title="VoiceSecure", layout="centered")

st.title("🎤 VoiceSecure - Panic Voice Detector")

st.write("Upload an audio file (.wav) to detect panic/distress level.")

# 📂 File uploader
uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

# 🚨 Alert logic
def trigger_alert(prediction, confidence):
    if prediction == "panic" and confidence > 0.75:
        return "🚨 HIGH ALERT"
    elif prediction == "distress" and confidence > 0.6:
        return "⚠️ Moderate Stress"
    elif confidence < 0.5:
        return "Uncertain"
    else:
        return "✅ Safe"

# 🎯 When file uploaded
if uploaded_file is not None:

    # Save file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # 🔥 Predict button
    if st.button("Analyze Audio"):
        with st.spinner("Analyzing..."):

            prediction, confidence = predict_audio(temp_path)

            confidence_score = round(confidence * 100, 2)
            alert = trigger_alert(prediction, confidence)

            # 🎯 OUTPUT
            st.subheader("🔍 Results")

            st.write(f"**Emotion:** {prediction}")
            st.write(f"**Confidence Score:** {confidence_score:.2f}%")
            st.write(f"**Status:** {alert}")

            # 🎨 Color alert
            if "HIGH ALERT" in alert:
                st.error("🚨 Emergency detected!")
            elif "Moderate" in alert:
                st.warning("⚠️ Stress detected")
            else:
                st.success("✅ Safe condition")

    # Cleanup (optional)
    if os.path.exists(temp_path):
        os.remove(temp_path)