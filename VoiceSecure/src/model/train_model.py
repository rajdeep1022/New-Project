import os
import numpy as np
import pickle

from scipy.io import wavfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ==========================
# DATASET PATH
# ==========================
DATASET_PATH = "dataset/processed_datasets"


# ==========================
# FEATURE EXTRACTION (SCIPY)
# ==========================
def extract_features(file_path):
    try:
        # Load audio
        sr, audio = wavfile.read(file_path)

        # Convert to float
        audio = audio.astype(np.float32)

        # Convert stereo → mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Normalize
        if np.max(np.abs(audio)) != 0:
            audio = audio / np.max(np.abs(audio))

        # Fixed length (1 sec = 16000 samples)
        max_len = 16000

        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]

        # Reshape into 2D (for CNN)
        feature = audio.reshape(160, 100)

        # Add channel dimension
        feature = feature[..., np.newaxis]

        return feature

    except Exception as e:
        print(f"❌ Error loading file: {file_path}")
        print(f"   Reason: {e}")
        return None


# ==========================
# LOAD DATASET
# ==========================
features = []
labels = []

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    print(f"\n📂 Processing folder: {label}")

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
print("\n✅ Total samples loaded:", len(features))

if len(features) == 0:
    raise ValueError("❌ No data loaded. Check dataset or audio files.")


X = np.array(features)
y = np.array(labels)


# ==========================
# ENCODE LABELS
# ==========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)


# ==========================
# SAVE ENCODER
# ==========================
os.makedirs("models", exist_ok=True)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


# ==========================
# TRAIN TEST SPLIT
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
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(160,100,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))


# ==========================
# COMPILE MODEL
# ==========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ==========================
# TRAIN MODEL
# ==========================
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)


# ==========================
# EVALUATE MODEL
# ==========================
loss, acc = model.evaluate(X_test, y_test)

print("\n🎯 Test Accuracy:", acc)


# ==========================
# SAVE MODEL
# ==========================
model.save("models/panic_voice_model.h5")

print("\n✅ Model saved successfully!")