import os
from datetime import datetime
from typing import Tuple, Any

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, \
    confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_DICT = {
        'rf': (RandomForestClassifier, 'RandomForest'),
        'xgb': (XGBClassifier, 'XGBoost'),
        'svm': (SVC, 'SupprotVectorMachine'),
        'gpc': (GaussianProcessClassifier, 'GaussianProcess')
}


def model_preparation(model_type: str, scale_pos_weight: float, grid_search: bool) -> Tuple[BaseEstimator, str]:
    """Prepare model with selected architecture.

    :param grid_search: Flag indicating that a grid search is to be performed or not.
    :param model_type: Type of model architecture selected to train.
    :param scale_pos_weight: The ratio of the number of negative samples to positive samples.
    :return: Model to train and model name for print/logs purposes.
    """

    try:
        model_class, model_name = MODEL_DICT[model_type]
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == 'rf':
        model = model_class(n_estimators=300, random_state=0, verbose=False, criterion='entropy',
                            class_weight=None, bootstrap=True, max_depth=None, min_samples_split=2,
                            min_samples_leaf=1)

    elif model_type == 'xgb':
        model = model_class(random_state=0, verbosity=2, scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
                            objective='binary:logistic', learning_rate=0.1, max_depth=3, n_estimators=100)

    elif model_type == 'svm':
        model = model_class(kernel='linear', C=1, probability=True, class_weight='balanced', degree=3, gamma='scale')

    elif model_type == 'gpc':
        model = model_class(random_state=0, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                            kernel=1.0 * RBF(1.0), max_iter_predict=50)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if grid_search:
        model = parameters_tuning(model, model_name)

    return model, model_name


def test_model(model, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, float | Any], Any]:
    """Calculate accuracy score of model.

    :param model: The trained model.
    :param x_test: Audio test data.
    :param y_test: Labels.
    :return: Dictionary with precision, recall and F1 metrics and average precision score separately.
    """
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred_proba = [round(x, 2) for x in y_pred_proba]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred,
                                                                           average='binary', zero_division='warn')
    average_precision = average_precision_score(y_test, y_pred)
    metrics = {'accuracy': round(acc, 4), 'precision': round(precision, 4),
               'recall': round(recall, 4), 'f1_score': round(f1_score, 4),
               'auprc': round(average_precision, 4), 'support': support}
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    conf_matrix = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    # Plot the precision-recall curve
    try:
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    except Exception as e:
        print(f"Error plotting precision-recall curve: {e}")

    return metrics, conf_matrix


def save_model(model: BaseEstimator, model_name: str, args: Any, precision: float) -> None:
    """Save a model to disk using joblib.

    Args:
        :param precision: Precision for the model on the test data.
        :param model: The trained model.
        :param args: Training parameters like model type, data chosen.
        :param model_name: Name of model architecture for path_name purposes.
    """
    # Get the directory of this file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Directory where to save the model.
    path_to_save_model = os.path.join(script_dir, args.path_to_save_model)

    # Create the directory if it doesn't exist
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    # Construct the path to save the model
    model_filename = f'{model_name}_{args.train_label}_prec:{precision:.3f}_{datetime.today().strftime("%Y-%m-%d")}.joblib'
    path = os.path.join(path_to_save_model, model_filename)

    # Save the model
    joblib.dump(model, path)


def parameters_tuning(model, model_name):
    if model_name == 'XGBoost':
        param_grid = {
                'learning_rate': [0.1, 0.01, 0.001],
                'max_depth': [3, 6, 9],
                'n_estimators': [100, 500, 1000],
                'objective': ['binary:logistic'],
                'eval_metric': ['logloss', 'error', 'auprc', 'map']
        }
    elif model_name == 'RandomForest':
        param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
        }
    elif model_name == 'GaussianProcess':
        param_grid = {
                'kernel': [RBF(), Matern(), RationalQuadratic()],
                'n_restarts_optimizer': [0, 1, 5],
                'max_iter_predict': [10, 50, 100, 200]
        }
    elif model_name == 'SupprotVectorMachine':
        param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
        }
    else:
        param_grid = {}

    # Create a StratifiedKFold object with 5 folds
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    # Create a GridSearchCV object
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)

    return grid
