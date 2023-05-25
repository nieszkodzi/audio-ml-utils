import argparse
import os

import joblib
import pandas as pd

import make_dataset
import models


def test_sample(model, args):
    """Evaluate model on single audio sample.

    :param model: Loaded model.
    :param args: Parser arguments with test label and key paths (path_to_audio, path_to_scaler).
    :return: File path with model prediction for the file.
    """
    # Prepare audio sample
    preprocessed_audio = make_dataset.prepare_sample(args.path_to_audio, args.path_to_scaler)
    prediction = model.predict(preprocessed_audio)
    print(f"Prediction for {args.path_to_audio} sample: {prediction}.")


def test_folder(model, args):
    """Evaluate model on multiple samples at given folder.

    :param model: Loaded model.
    :param args: Parser arguments with label to test and key paths.
    :return: Path of every file with model prediction for it.
    """
    data = []
    for filename in os.listdir(args.path_to_audio):
        path_to_audio = os.path.join(args.path_to_audio, filename)
        if os.path.isfile(path_to_audio):
            preprocessed_audio = make_dataset.prepare_sample(path_to_audio, args.path_to_scaler)
            prediction = model.predict(preprocessed_audio)
            print(f"Prediction for {filename} sample: {prediction[0]}.")

            # create a dictionary with the data
            file_data = {"filename": filename,
                         "instrument_piano": prediction[0]}

            # add the dictionary to the list
            data.append(file_data)
    df = pd.DataFrame(data)

    # Get the directory path of the audio folder
    audio_folder = os.path.dirname(args.path_to_audio)

    # Construct the path to the predictions.csv file
    predictions_file = os.path.join(audio_folder, '..', 'predictions.csv')

    # Save the DataFrame to the predictions.csv file
    df.to_csv(predictions_file, index=False)


def test_dataset(model, args):
    """Evaluate model on dataset given by csv file with file paths and labels.

    :param model: Loaded model.
    :param args: Parser arguments with label to test and key paths.
    :return: Results of the evaluation at this dataset,
    f1 score, precision, recall and area under precision-recall curve.
    """
    if not args.path_to_csv:
        raise ValueError('No path to test dataset metadata given.')
    # Prepare test dataset
    test_audio, test_labels = make_dataset.test_dataset(args.path_to_audio, args.path_to_csv, args, scaler=None)
    metrics, confusion_matrix = models.test_model(model, test_audio, test_labels)

    print("Confusion matrix")
    print(confusion_matrix)
    print(f"F1-Score: {metrics['f1_score']} - "
          f"Precision: {metrics['precision']} - "
          f"Recall: {metrics['recall']}")
    print(f"Area Under Precision-Recall Curve {metrics['auprc']}")
    print(f"Accuracy: {metrics['accuracy']}")


def main():
    parser = argparse.ArgumentParser(description='Test a machine learning model')
    parser.add_argument('--test', type=str, required=True,
                        help='Choose what you want to classify: "sample", "folder" or "dataset". '
                             'If you want to classify hole dataset, csv path required.')
    parser.add_argument('--path_to_model', type=str, required=True,
                        help='Path to joblib model.')
    parser.add_argument('--path_to_scaler', type=str, required=True,
                        help='Path to joblib scaler.')
    parser.add_argument('--path_to_csv', type=str,
                        help='Path to metadata file')
    parser.add_argument('--path_to_audio', type=str, required=True,
                        help='Path to audio folder or single file.')
    parser.add_argument('--train_label', type=str, required=True,
                        help='The label for which the model was trained and now we want to test it.'
                             'Write label as strings with "_" between words, '
                             'ex. "instrument_piano"  or "instrument_violin"')
    args = parser.parse_args()

    # Load the model
    model = joblib.load(args.path_to_model)

    if args.test == 'dataset':
        test_dataset(model, args)

    elif args.test == 'sample':
        test_sample(model, args)

    elif args.test == 'folder':
        test_folder(model, args)

    else:
        raise ValueError('You need to choose dataset or sample as --test argument, '
                         'regarding to what you want to predict.')


if __name__ == '__main__':
    main()
