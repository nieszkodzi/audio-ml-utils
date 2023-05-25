<h1 align="center">
  <br>
  Natalia's Machine Learning Utils
</h1>

<p align="center">
  <a href="#key-features">Introduction</a> •
  <a href="#how-to-use">Getting Started</a> •
  <a href="#included">Understand files</a> •
</p>

# Introduction

Repository to train & test AI and ML models.

# Getting Started

1. Clone repository by SSH if you created and added your public key:

```bash
$ git clone git@github.com:nieszkodzi/ml-utils.git
```

2. Create virtual environment.

```bash
$ python3 -m venv ./ml-utils/venv
```

3. Install dependencies.

```bash
$ pip install -r requirements.txt
```

# Understand files

[experiments/train.py](experiments/train.py)

script to train models,

**required parameters:** 'path_to_audio', 'path_to_csv', 'model_type', 'train_label',

**additional_parameters:** 'path_to_save_model', 'path_to_test_audio', 'path_to_test_csv', 'data_frac', '
negatives_to_positives_frac'

**usage example:**
*python3 experiments/train.py --path_to_audio '/home/{user}/Data/train_audio/' --path_to_test_audio '
/home/{user}/Data/test_audio' --train_label 'piano' --data_frac 1 --negatives_to_positives_frac
1 --model_type 'gpc' --path_to_csv '/home/{user}/Data/metadata.csv' --path_to_test_csv '
/home/{user}/Data/test_metadata.csv'*
___

[experiments/test.py](experiments/test.py)

script to test model,

**required parameters:** 'test', 'path_to_audio', 'path_to_model', 'path_to_scaler', 'train_label', 'path_to_csv'

**usage example:**
*python3 experiments/test.py --test 'dataset' --path_to_audio '/home/{user}/Data/train_audio/' --path_to_model "
saved_models/XGBoost_f1:0.833_2023-02-23.joblib" --path_to_scaler "saved_models/scaler_xgb_2023-02-23.joblib"  --train_label 'piano'
--path_to_csv '/home/{user}/Data/test_metadata.csv'*
___

[experiments/make_dataset.py](experiments/make_dataset.py)

additional code used by train.py and test.py to create train or test datasets; feature extraction, splitting, cross
validation etc.
___

[experiments/models.py](experiments/models.py)

additional code used by train.py and test.py with preparation of models to train and test and saving trained models
___

[experiments/visualization.py](experiments/visualization.py)

code to visualize audio features such as mfcc, wave forms etc. to choose the best candidates to training and code to
extract audioset pretrained embeddings to improve models.
___

[utils/audio_utils.py](utils/audio_utils.py)

contains functions to convert m4a to wav (more will be added when needed), extract additional features
___

[utils/python_utils.py](utils/python_utils.py)

by now contains only function to extract csv out of audio folder
