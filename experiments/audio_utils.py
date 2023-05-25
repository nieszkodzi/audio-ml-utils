import argparse
import os
import subprocess

import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

FRAME_LENGTH = 1024
HOP_LENGTH = 512
N_MFCC = 40
N_MELS = 20


def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    # Load the MP3 file using pydub
    audio = AudioSegment.from_mp3(mp3_file_path)

    # Export the audio as a WAV file using pydub
    audio.export(wav_file_path, format='wav')


def convert_m4a_to_wav():
    """Convert m4a files to wav ones. It takes hole folder as an argument
    and creates copies of files with new extension.

    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Convert m4a files to wav format.')
    parser.add_argument('--dir', metavar='dir', type=str, help='Directory path containing m4a files to be converted')

    # Parse the arguments
    args = parser.parse_args()

    # Loop through all the files in the directory
    for file in os.listdir(args.dir):
        print(os.listdir(args.dir))
        if file.endswith('.m4a'):
            # Get the input and output file paths
            input_path = os.path.join(args.dir, file)
            print(input_path)
            output_path = os.path.join(args.dir, os.path.splitext(file)[0] + '.wav')

            # Use ffmpeg to convert the file
            subprocess.run(['ffmpeg', '-i', input_path, output_path])


def extract_audioset_embeddings(audio_file: str):
    """Pre-trained AudioSet models used to extract embeddings from audio file
     and then these embeddings as features for a classification model.

    :param audio_file: path to audio file.
    :return: extracted embeddings.
    """
    # Load AudioSet embedding model
    embedding_model = tf.keras.models.load_model('audioset_embedding_model.h5')

    # Convert audio to spectrogram
    audio, sr = librosa.load(audio_file, sr=22050)
    spec = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag_spec = np.abs(spec)
    log_mag_spec = librosa.amplitude_to_db(mag_spec)

    # Pad spectrogram to desired shape
    target_shape = (128, 128)
    spec_shape = log_mag_spec.shape
    if spec_shape[1] < target_shape[1]:
        padding = ((0, 0), (0, target_shape[1] - spec_shape[1]))
        log_mag_spec = np.pad(log_mag_spec, padding, 'constant')
    else:
        log_mag_spec = log_mag_spec[:, :target_shape[1]]

    # Add batch dimension
    log_mag_spec = np.expand_dims(log_mag_spec, axis=0)

    # Extract AudioSet embeddings
    embeddings = embedding_model.predict(log_mag_spec)

    return embeddings


def extract_features(audio, sr):
    """The feature extraction function: spectral centroid, spectral flux and zero crossing rate.

    :param sr:
    :param audio:
    :return:
    """

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)

    # Logarithm of spectrogram
    spec = librosa.stft(audio)
    log_spec = librosa.amplitude_to_db(abs(spec))

    # Extract RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    # Extract zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    # Extract spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

    # Extract spectral flux
    spec_flux = librosa.onset.onset_strength(y=audio, sr=sr)

    return mfcc, mel, log_spec, rms, zcr, spec_centroid, [spec_flux]


def apply_highpass_filter(audio, sr, cutoff_freq=6000):
    # Define filter parameters
    order = 4

    # Calculate filter coefficients
    nyquist_freq = 0.5 * sr
    normal_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff_freq, btype='low', analog=False)

    # Apply filter to audio signal
    filtered_audio = filtfilt(b, a, audio)

    return filtered_audio


if __name__ == '__main__':
    mp3_file_path = '/home/natalia/Downloads/untitled.mp3'
    wav_file_path = '/home/natalia/Data/raw_example.wav'
    convert_mp3_to_wav(mp3_file_path, wav_file_path)
