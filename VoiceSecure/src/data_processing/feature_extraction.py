import numpy as np
from scipy.io import wavfile
from scipy.fft import fft

def extract_features(file_path):
    try:
        sr, audio = wavfile.read(file_path)

        # mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        features = []

        # 🔹 Time domain
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.max(audio),
            np.min(audio),
            np.median(audio),
            np.percentile(audio, 25),
            np.percentile(audio, 75)
        ])

        # 🔹 Energy
        features.append(np.sum(audio**2))

        # 🔹 Zero crossing rate
        features.append(np.mean(np.abs(np.diff(np.sign(audio)))))

        # 🔥 FFT features (VERY IMPORTANT)
        fft_vals = np.abs(fft(audio))[:len(audio)//2]

        features.extend([
            np.mean(fft_vals),
            np.std(fft_vals),
            np.max(fft_vals),
            np.percentile(fft_vals, 25),
            np.percentile(fft_vals, 75)
        ])

        # 🔥 Chunk-based features (MORE GRANULAR)
        chunks = np.array_split(audio, 10)

        for chunk in chunks:
            features.append(np.mean(chunk))
            features.append(np.std(chunk))
            features.append(np.max(chunk))

        return np.array(features)

    except:
        return None