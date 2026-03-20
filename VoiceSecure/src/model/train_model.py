import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# IMPORT YOUR MODULES
from src.data_processing.feature_extraction import extract_features
from src.model.model_architechture import build_model


# ==========================
# DATASET PATH
# ==========================
DATASET_PATH = "dataset/processed_datasets"


# ==========================
# LOAD DATASET
# ==========================
features = []
labels = []

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    print(f"\n📂 Processing: {label}")

    for file in os.listdir(folder):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder, file)

        feature = extract_features(file_path)

        if feature is not None:
            features.append(feature)
            labels.append(label)


# ==========================
# CHECK DATA
# ==========================
print("\n✅ Total samples:", len(features))

if len(features) == 0:
    raise ValueError("No data loaded!")


X = np.array(features)
y = np.array(labels)


# ==========================
# FEATURE SCALING
# ==========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
os.makedirs("models", exist_ok=True)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# ==========================
# ENCODE LABELS
# ==========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# Save encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


# ==========================
# TRAIN-TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)


# ==========================
# CLASS WEIGHTS
# ==========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)

class_weights = dict(enumerate(class_weights))


# ==========================
# BUILD MODEL
# ==========================
model = build_model(X.shape[1])


# ==========================
# COMPILE
# ==========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================
# TRAIN
# ==========================
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)


# ==========================
# EVALUATE
# ==========================
loss, acc = model.evaluate(X_test, y_test)

print("\n🎯 Accuracy:", acc)


# ==========================
# SAVE MODEL
# ==========================
model.save("models/panic_voice_model.h5")

print("\n✅ Model saved successfully!")