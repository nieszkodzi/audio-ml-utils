import os
from datetime import datetime
from typing import Tuple, Any

import joblib
import librosa
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio_utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_LENGTH = 1024
HOP_LENGTH = 512
N_MFCC = 30
N_MELS = 30
SAMPLE_LENGTH = 4410


def extract_features(audio_sample_path: str, start: int, end: int, low_pass_filter=False) -> np.ndarray:
    """Load audio file by librosa and calculate MFCC.

    :param low_pass_filter:
    :param start: start of the sound
    :param end: end of the sound
    :param audio_sample_path: Path to audio file.
    :return: Mean of features.
    """
    audio, sr = librosa.load(audio_sample_path, sr=44100)
    if end != 0:
        if audio.shape[0] > start + SAMPLE_LENGTH:
            end = start + SAMPLE_LENGTH
            audio = audio[start:end]
        else:
            audio = audio[audio.shape[0] - SAMPLE_LENGTH:]
    else:
        # Trim the silence
        non_silent_audio, _ = librosa.effects.trim(y=audio, top_db=30)
        audio = non_silent_audio[:SAMPLE_LENGTH]

    if low_pass_filter:
        audio = audio_utils.apply_highpass_filter(audio, sr, cutoff_freq=8000)

    # Extract audio features
    mfcc, mel, log_spec, rms, zcr, spec_centroid, spec_flux = audio_utils.extract_features(audio, sr)

    # Concatenate features
    features = np.concatenate((mfcc, mel, spec_centroid, rms, zcr, [spec_flux]), axis=0).flatten()

    return features


def prepare_sample(audio_sample_path: str, path_to_scaler: str) -> np.ndarray:
    """Take one sample, extract features and scale.

    :param audio_sample_path: Path to audio file.
    :param path_to_scaler: Path to scaler saved during the training.
    :return: Preprocessed audio.
    """
    x = extract_features(audio_sample_path, start=0, end=0)
    x = np.array(x)
    scaler = joblib.load(path_to_scaler)
    preprocessed_audio = scaler.transform(x.reshape(1, -1))

    return preprocessed_audio


def clean_data(path_to_audio: str, data: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe out of missing audio files.

    :param path_to_audio: Path to folder with audio files.
    :param data: Dataframe with metadata.
    :return: Metadata dataframe without missing files.
    """
    # Remove non-existing files from data frame.
    for index, row in data.iterrows():
        audio_path = os.path.join(path_to_audio, row['filename'])
        if not os.path.exists(audio_path):
            data = data[data['filename'] != row['filename']]

    return data


def calc_samples_weight(y, strength_lvl):
    """The more 'powerful' sample is the bigger impact it has for training.

    :param strength_lvl: list of level of strength for each sample
    :param y: Labels
    :return: Weight of samples regarding of 'power' of positive sample
    """

    # Map strength values to weights
    strength_to_weight = {'low': 0.5, 'medium': 1.0, 'strong': 2.0, 'unknown': 1.0}
    weights = strength_lvl.map(strength_to_weight)

    # Set weights to 1 for negative samples
    weights[y == 0] = 1.0

    return weights


def sklearn_dataset(args, path_to_audio, data: pd.DataFrame, scaler, train: bool) -> \
        tuple[Series | DataFrame | Any, Any, Any, Any, Any]:
    """Prepare dataset to be used with sklearn models.

    :param scaler: Normalization function for data.
    :param path_to_audio: Folder where audio files are stored.
    :param train: Flag, if true then we prepare train dataset, if false - test one.
    :param data: Dataframe with metadata.
    :param args: Parser arguments with instrument to detect, model type and key paths.
    :return: Dataset split to train and validation.
    """

    column_name = 'instrument_' + args.train_label

    # Split data into 0 and 1 classes, avoid duplicates
    class_1 = data[data[column_name] == 1]
    negative_data = data[data[column_name] == 0]
    negative_data = negative_data.drop_duplicates(subset=['filename'], keep='first')
    mask = ~negative_data['filename'].isin(class_1['filename'])
    class_0 = negative_data[mask]

    if train is True:
        # Randomly sample a subset of class 0 data
        negatives_amount = int(args.negatives_to_positives_frac * len(class_1))
        if len(class_0) > negatives_amount:
            class_0_sample = class_0.sample(n=negatives_amount, random_state=0)

            # Concatenate class 0 sample and class 1 data
            sampled_data = pd.concat([class_0_sample, class_1])

            # Shuffle the data
            data = sampled_data.sample(frac=1, random_state=0)

    x = []
    for idx, row in tqdm(data.iterrows()):
        features = extract_features(os.path.join(path_to_audio, row['filename']), row['start'], row['end'])
        x.append(features)

    y = data[column_name].values

    if train is True:
        scaler = StandardScaler()
        scaler.fit(x)
        if args.path_to_save_model:
            scaler_path = os.path.join(SCRIPT_DIR, args.path_to_save_model,
                                       f'scaler_{args.model_type}_{datetime.today().strftime("%Y-%m-%d")}.joblib')
            joblib.dump(scaler, scaler_path)
    else:
        if scaler is None:
            scaler = joblib.load(args.path_to_scaler)
        else:
            pass
    x = scaler.transform(x)

    artist = data['artist'].values

    return data, x, y, artist, scaler


def train_dataset(args) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Prepare train dataset.

    :param args: Parser arguments with instrument to detect, model type and key paths.
    :return: Two, separated arrays with audio data for train and test purposes
    and two others with train and test labels.
    """
    data = pd.read_csv(os.path.join(SCRIPT_DIR, args.path_to_csv))

    # Leave only existing files
    data = clean_data(args.path_to_audio, data)

    # Select only a given fraction of data
    data = data.sample(frac=args.data_frac, random_state=0)

    data, x, y, artist, scaler = sklearn_dataset(args, args.path_to_audio, data, scaler=None, train=True)
    test_artists = []
    train_artists = []

    # Create the splitter object
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    # Split the data into training and test sets
    for train_index, test_index in gss.split(data, groups=artist):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        data.reset_index(inplace=True)
        for index in test_index:
            test_artists.append(data.loc[index, 'artist'])
        for index in train_index:
            train_artists.append(data.loc[index, 'artist'])
        train_strength_lvl = data.loc[train_index, 'strength_lvl']

    print(f"Training of {args.model_type} with instrument {args.train_label}, "
          f"negatives/positives = {args.negatives_to_positives_frac}")

    print('artists in train dataset: ', np.unique(np.array(train_artists)))
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print(print(f"Unique labels in train dataset: {unique_labels} and their counts {counts}."))

    print("------------------------------------------------------")

    print('artists in test dataset: ', np.unique(np.array(test_artists)))
    unique_labels, counts = np.unique(y_test, return_counts=True)
    print(print(f"Unique labels in test dataset: {unique_labels} and their counts {counts}."))

    print('Weights of samples calculated.')
    train_samples_weight = calc_samples_weight(y_train, train_strength_lvl)

    return x_train, x_test, y_train, y_test, train_samples_weight, scaler


def test_dataset(path_to_audio, path_to_csv, args, scaler) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare test dataset.

    :param args: Parser arguments with instrument to detect, model type and key paths.
    :return: Array with audio data and another one with labels.
    """
    data = pd.read_csv(os.path.join(SCRIPT_DIR, path_to_csv))
    data = clean_data(path_to_audio, data)
    _, x, y, _, _ = sklearn_dataset(args, path_to_audio, data, scaler, train=False)

    return x, y
