import argparse
# import os
# import pandas as pd
from datetime import datetime
from hp_tuning import HypeTuner
from models.tabnet2_reg_model_fit import NNModelTrain
from models.xgb_reg_model_fit import XGBModelTrain
# from models.hybrid_reg_model_fit import HYBRIDModelTrain


def main(args):
    """
    Executes the main pipeline for model training, hyperparameter tuning, and
    optional plotting. The script supports both XGBoost and Neural Network
    model types, allowing for hyperparameter tuning and evaluation.

    This function performs the following tasks:
    - Initializes data file and model parameters for training.
    - Checks command-line arguments to determine tasks like plotting or
      hyperparameter tuning.
    - Creates and tunes models based on the chosen algorithm (XGBoost
      or Neural Network).
    - Runs the model training pipeline after optionally identifying the best
      hyperparameters through cross-validation.

    The final model and its results will be stored in specified files, and
    execution time will be printed.

    :param args: Parsed command-line arguments with attributes corresponding
        to available options (e.g., `tune`, `xgb`, `nn`).
        Command-line flags determine the operational flow of the script.
    :type args: argparse.Namespace
    :return: None
    """
    t0 = datetime.now()
    train_data_file = 'data/nyc-taxi-trip-duration/train_processed.csv'
    test_data_file = 'data/nyc-taxi-trip-duration/test_processed.csv'
    best_params = {}
    prefix = ''
    if args.tune:
        prefix = 'hp_'
        if args.xgb:
            trainer = XGBModelTrain(
                train_data_file,
                test_data_file,
                'models/hp_xgbreg.xz'
            )
            model_params = {
                'max_depth': [2, 5, 10],
                'learning_rate': [1e-01, 1e-02, 1e-03],
                'n_estimators': [500, 750],
                'reg_alpha': [0.01, 0.1, 0],
                'reg_lambda': [0.01, 0.1, 0],
            }
        if args.nn:
            trainer = NNModelTrain(
                train_data_file,
                test_data_file,
                'models/hp_tabnetreg.xz'
            )
            model_params = {
                "n_d": [8, 16, 32],
                "gamma": [1.3, 1.7],
                "n_shared": [2, 4],
                "lambda_sparse": [1e-3, 1e-4],
            }
        tuner = HypeTuner(trainer, model_params)
        tuner.pipeline()
        print(tuner.results.to_markdown())
        best_params = tuner.get_best_params()
        print(best_params)
    # Overwrite trainer with
    # a fresh instance
    if args.xgb:
        trainer = XGBModelTrain(
            train_data_file,
            test_data_file,
            f'models/{prefix}xgbreg.xz',
        )
    if args.nn:
        trainer = NNModelTrain(
            train_data_file,
            test_data_file,
            f'models/{prefix}tabnetreg.xz',
        )
    if best_params != {}:
        trainer.ai_model.model_params = best_params
    trainer.pipeline()
    t1 = datetime.now()
    print(f"Script `{__file__}` execution time: {t1-t0}")
    return

def parse_args():
    """
    Parses command line arguments for training and tuning machine learning models.

    This function creates a command line argument parser for configuring machine
    learning model training. It supports mutually exclusive model selection between
    XGBoost regressor and neural network, as well as an additional option for tuning.
    Users must select one model to train.

    :return: Parsed command line arguments as a namespace object.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb", action="store_true", help="Train xgboost regressor")
    parser.add_argument("--nn", action="store_true", help="Train tabnet neural network")
    parser.add_argument("--tune", action="store_true", help="Tune the model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)