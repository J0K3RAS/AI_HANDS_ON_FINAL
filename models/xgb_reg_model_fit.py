"""
Created on 5 April 2025
@authors: Charalampos

Example Usage:
1) Train the XGBoost regression model for taxi trip prediction
    python xgb_reg_model_fit.py
        --train-file <Path to train .csv>
        --test-file <Path to test .csv>
        --model-file <Path to model>
"""
import argparse
import warnings
from datetime import datetime
from models.xgb_reg import XGBModel
from models.templates.ai_train_template import Trainer

warnings.filterwarnings('ignore')

class XGBModelTrain(Trainer):
    """
        Implements the Trainer abstract class, for
        XGBoost regression model.
    """
    def __init__(self, train_file, test_file, model_file):
        """
            Constructor of class XGBModelTrain.

            :param train_file: str, filepath of train .csv
        """
        super().__init__(train_file, test_file, model_file)
        self.ai_model = XGBModel()

    @property
    def x_colnames(self):
        """
            List of x-column names used by the model.
        """
        return self.ai_model.x_colnames

    @property
    def y_colname(self):
        """
            String of y-column name
        """
        return self.ai_model.y_colname


    def load_model(self):
        """
            Load the model from model_file and return
            a new Trainer class instance, with attribute
            the loaded model.
        """
        model = self.ai_model.load_model(self.model_file)
        other = XGBModelTrain(self._train_file, self._test_file, self.model_file)
        other.ai_model = model
        return other

    def save_model(self, path):
        """
            Replace the current model, if it
            is better than the old one.

            :param path: str, the save path
        """
        self.ai_model.save_model(path)
        return self

    def predict(self, x):
        """
            Use the current model to make
            a prediction for x-data.

            :param x: pd.DataFrame, x-data
        """
        return self.ai_model.predict(x)

    def train_model(self):
        """
            Using train DataFrame, train the
            XGBoost regression model.
        """
        print('Model initialization & training')
        self.ai_model.train_model(
            self.train_df[self.x_colnames],
            self.train_df[self.y_colname],
            self.val_df[self.x_colnames],
            self.val_df[self.y_colname]
        )
        # Printing metrics
        perf = self.evaluate_model()
        print(f'Model performance: \n{perf.to_markdown()}')
        return self

    def pipeline(self):
        """ Driver method of this class """
        self.train_model().save_model(path=self.model_file)


def parse_input(args=None):
    """
       Parse cmd line arguments
       :param args: The command line arguments provided by the user
       :return: The parsed input Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--train-file", type=str, action="store", metavar="data_file",
                        required=True, help="Path to csv file with data to train the model on.")
    parser.add_argument("-t", "--test-file", type=str, action="store", metavar="test_file",
                       required=True, help="Path to csv file with test data.")
    parser.add_argument("-m", "--model-file", type=str, action="store", metavar="model_file",
                        required=True, help="The sav to store the newly trained model")

    return parser.parse_args(args)


def main(args):
    """ The main method """
    start_time = datetime.now()

    re_trainer = XGBModelTrain(
        train_file=args.train_file,
        test_file=args.test_file,
        model_file=args.model_file
    )
    re_trainer.pipeline()

    print(f"Script execution time: {datetime.now() - start_time}")
    return re_trainer


if __name__ == '__main__':
    ARG = parse_input()
    re_train = main(ARG)
