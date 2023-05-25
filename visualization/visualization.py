import argparse

import librosa.display
import numpy as np
from matplotlib import pyplot as plt

FRAME_LENGTH = 2048
HOP_LENGTH = 512


def show_plots(audio_path: str):
    """Draw a waveform or waveform rate of change out of audio file.

    :param audio_path: path to audio file.
    """
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Find the start and end times of non-silent sections

    # Trim the silence
    y_trimmed, idx = librosa.effects.trim(y, top_db=20, frame_length=1024, hop_length=256)

    # Define the amount of audio to save on each side of the trimmed audio
    margin_left = 0.1  # in seconds
    margin_right = 0.1  # in seconds

    # Get the start and end indices of the portion to be extracted
    start_idx = max(idx[0] - int(margin_left * sr), 0)
    end_idx = min(idx[1] + int(margin_right * sr), len(y))

    # Extract the desired portion of the audio
    y = y[start_idx:end_idx]
    y_scale = librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

    # Extract audio features
    waveform = y
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_spec = librosa.power_to_db(np.abs(y_scale) ** 2)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr)

    # Plot waveform and rate of change subplot
    plt.figure(figsize=(20, 10))

    # Plot waveform
    plt.subplot(2, 4, 1)
    librosa.display.waveshow(waveform, sr=sr)
    plt.title('Waveform')

    # Plot rate of change
    plt.subplot(2, 4, 2)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='linear')
    plt.title('MFCC')

    # Plot rate of change
    plt.subplot(2, 4, 3)
    librosa.display.specshow(log_spec, sr=sr, x_axis='time', y_axis='log', hop_length=HOP_LENGTH)
    plt.title('Log-Spectrogram')

    # Plot rate of change
    plt.subplot(2, 4, 4)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel-Spectrogram')

    # Plot rate of change
    plt.subplot(2, 4, 5)
    plt.plot(centroid)
    plt.title('Spectral centroid')

    # Plot rate of change
    plt.subplot(2, 4, 6)
    plt.plot(flux)
    plt.title('Spectral flux')

    # Plot rate of change
    plt.subplot(2, 4, 7)
    plt.plot(zcr[0])
    plt.title('Zero crossing rate')

    # Plot rate of change
    plt.subplot(2, 4, 8)
    plt.plot(rms[0])
    plt.title('RMS Energy')

    plt.show()


if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Visualize soundwave of an audio file')
    parser.add_argument('--audio_path', help='Path to the audio file')

    # Parse the arguments
    args = parser.parse_args()

    show_plots(args.audio_path)
