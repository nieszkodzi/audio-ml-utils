import argparse
import os

import neptune
import neptune.integrations.sklearn as npt_utils

import make_dataset
import models


def print_results(metrics, confusion_matrix, train_or_test):
    """Prints results of model evaluation and logs them with Neptune.ai.

    :param metrics: F1-score, precision, recall, Area Under Precision-Recall Curve and accuracy.
    :param confusion_matrix: Results of confusion matrix, but in the form of 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp.
    :param train_or_test: Specifies whether the results are for the train or test dataset.
    :return: None (only prints and logs)
    """
    # Print important metrics.
    print("Confusion matrix results")
    print(confusion_matrix)
    print(f"F1-Score: {metrics['f1_score']} - "
          f"Precision: {metrics['precision']} - "
          f"Recall: {metrics['recall']}")
    print(f"Area Under Precision-Recall Curve {metrics['auprc']}")
    print(f"Accuracy: {metrics['accuracy']}")

    # Save logs by Neptune.ai
    run[f"logs/{train_or_test}/f1_score"] = metrics['f1_score']
    run[f"logs/{train_or_test}/precision"] = metrics['precision']
    run[f"logs/{train_or_test}/recall"] = metrics['recall']
    run[f"logs/{train_or_test}/auprc"] = metrics['auprc']
    run[f"logs/{train_or_test}/accuracy"] = metrics['accuracy']


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a machine learning model')
    parser.add_argument('--model_type', type=str, default='xgb',
                        help='Type of model to train: "rf" for Random Forest, "xgb" for XGBoost, "gpc" for Gaussian '
                             'Process Classifier or "svm" for Support Vector Machine')
    parser.add_argument('--path_to_csv', type=str, default=os.path.join('../data/metadata.csv'),
                        help='Path to metadata csv file.')
    parser.add_argument('--path_to_audio', type=str, required=True,
                        help='Path to audio folder.')
    parser.add_argument('--path_to_test_csv', type=str,
                        help='Path to metadata csv file.')
    parser.add_argument('--path_to_test_audio', type=str,
                        help='Path to audio folder.')
    parser.add_argument('--data_frac', type=float, default=1.0,
                        help='Fraction of samples took from metadata file')
    parser.add_argument('--negatives_to_positives_frac', type=float, default=1.0,
                        help='Fraction of negative samples to positive samples')
    parser.add_argument('--train_label', type=str, required=True,
                        help='Label to train. Write label as strings, ex. "piano"  or "violin"')
    parser.add_argument('--path_to_save_model', type=str, default=os.path.join('../saved_models'),
                        help='Path to the folder where to save the model.')
    parser.add_argument('-g', '--grid_search', action='store_true',
                        help='Do GridSearch.')
    args = parser.parse_args()

    # Track dataset by Neptune.ai
    run["dataset/csv"].track_files(args.path_to_csv)
    run["dataset/full"].upload(args.path_to_csv)
    run["model"] = args.model_type

    # Prepare data
    x_train, x_valid, y_train, y_valid, train_samples_weight, scaler = make_dataset.train_dataset(args)

    # Choose the model based on the command line argument
    model, model_name = models.model_preparation(args.model_type, args.negatives_to_positives_frac, args.grid_search)

    # Train the model
    weighting_models = ['rf', 'xgb', 'svm']
    if args.model_type in weighting_models:
        model.fit(x_train, y_train, sample_weight=train_samples_weight)
    else:
        print("Sample weighting NOT used - this model does not use it.")
        model.fit(x_train, y_train)

    if args.grid_search:
        # Get the best estimator
        model = model.best_estimator_
        # Print best params
        print("Best parameters:")
        print(model.get_params())

    # Test the model
    print("--------------------------------------------------------------------")
    print("VALID DATASET RESULTS")
    metrics, confusion_matrix = models.test_model(model, x_valid, y_valid)
    print_results(metrics, confusion_matrix, 'train')

    # Save logs by Neptune.ai
    run["model_summary"] = npt_utils.create_classifier_summary(model, x_train, x_valid, y_train, y_valid)

    if args.path_to_test_audio:
        print("--------------------------------------------------------------------")
        print("TEST DATASET RESULTS")
        x_test, y_test = make_dataset.test_dataset(args.path_to_test_audio, args.path_to_test_csv, args, scaler)
        metrics, confusion_matrix = models.test_model(model, x_test, y_test)
        print_results(metrics, confusion_matrix, 'test')

    # Save the model
    if args.path_to_save_model:
        models.save_model(model, model_name, args, metrics['precision'])


if __name__ == '__main__':
    run = neptune.init_run()
    main()
    run.stop()
