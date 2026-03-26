from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from predict import predict_panic


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    print("📁 File received:", file.filename)

    result = predict_panic(file_path)

    print("🧠 Prediction:", result)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)