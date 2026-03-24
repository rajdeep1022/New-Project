from scipy.io import wavfile
import numpy as np

def process_audio(file_path):
    sample_rate, data = wavfile.read(file_path)

    # Convert to float
    data = data.astype(np.float32)

    # Normalize
    data = data / np.max(np.abs(data))

    # Fix length (example: 16000 samples = 1 sec)
    max_length = 16000

    if len(data) > max_length:
        data = data[:max_length]
    else:
        pad_width = max_length - len(data)
        data = np.pad(data, (0, pad_width), mode='constant')

    # Reshape for model
    data = data.reshape(1, -1)

    return data