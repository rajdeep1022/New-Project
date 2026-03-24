import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from src.data_processing.feature_extraction import extract_features
from src.model.model_architechture import build_model

DATASET_PATH = "dataset/processed_datasets"

features = []
labels = []

print("🚀 Loading dataset...")

# 🔹 Load dataset
for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    print(f"📂 Processing: {label}")

    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(folder, file)

            feat = extract_features(file_path)

            if feat is not None:
                features.append(feat)
                labels.append(label)

# Convert
X = np.array(features)
y = np.array(labels)

print("✅ Data shape:", X.shape)

# 🔹 Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 🔹 Shuffle (IMPORTANT)
X, y_categorical = shuffle(X, y_categorical, random_state=42)

# 🔹 Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Save scaler & encoder
os.makedirs("models", exist_ok=True)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# 🔹 Class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))

# 🔹 Build model
model = build_model(X.shape[1], y_categorical.shape[1])

# 🔹 Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 🔹 Train
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# 🔹 Save model
model.save("models/panic_voice_model.h5")

print("🎉 Training Complete!")