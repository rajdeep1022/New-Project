import numpy as np
from scipy.io import wavfile
from scipy.fft import fft

def extract_features(file_path):
    try:
        sr, audio = wavfile.read(file_path)

        audio = audio.astype(np.float32)

        # mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # normalize
        if np.max(np.abs(audio)) != 0:
            audio = audio / np.max(np.abs(audio))

        # ===== TIME DOMAIN FEATURES =====
        mean = np.mean(audio)
        std = np.std(audio)
        max_val = np.max(audio)
        min_val = np.min(audio)

        # ===== FREQUENCY DOMAIN =====
        fft_vals = np.abs(fft(audio))[:len(audio)//2]

        fft_mean = np.mean(fft_vals)
        fft_std = np.std(fft_vals)

        # ===== ENERGY =====
        energy = np.sum(audio**2)

        # Combine all features
        feature = np.array([
            mean, std, max_val, min_val,
            fft_mean, fft_std,
            energy
        ])

        return feature

    except Exception as e:
        print("Error:", e)
        return None