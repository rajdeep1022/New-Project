from utils.feature_extraction import extract_features
import numpy as np

def process_audio(file_path):
    try:
        features = extract_features(file_path)
        features = np.array(features)
        features = features.reshape(1, -1)

        print("Feature shape:", features.shape)

        return features

    except Exception as e:
        raise Exception(f"Feature extraction error: {str(e)}")