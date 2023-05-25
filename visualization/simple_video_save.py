# import subprocess
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

image_path = '/home/natalia/Data/spectrogram.jpg'
sound_path = '/home/natalia/Data/sample.wav'
output_path = '/home/natalia/Data/video.mp4'

# Usage:
# subprocess.call(
#         ['ffmpeg', '-loop', '1', '-i', image_path, '-i', sound_path, '-c:v', 'libx264', '-c:a', 'aac', '-strict',
#          'experimental', '-b:a', '192k', '-t', '1', output_path])

# Load audio file using librosa
y, sr = librosa.load(sound_path)

# Convert the spectrogram to decibels (dB)
log_spec = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

# Plot and save spectrogram as JPG
librosa.display.specshow(log_spec, sr=sr, x_axis='time', y_axis='mel')
plt.title('Log-Scaled Spectrogram of {}'.format(sound_path))
plt.tight_layout()

plt.savefig('/home/natalia/Data/spectrogram.jpg')
