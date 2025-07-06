import itertools as it
import pandas as pd
from contextlib import closing
from functools import partial
# from memory_profiler import profile
from multiprocessing import Pool
# from sklearn.model_selection import KFold

class HypeTuner:
    """
    Facilitates hyperparameter tuning for machine learning models.

    The `HypeTuner` class is designed to optimize the performance of a machine learning
    model by testing different hyperparameter combinations based on the given parameter
    grid. The tuning process evaluates the combinations using cross-validation and
    aggregates the performance metrics to identify the best parameter set.

    :ivar trainer: The object responsible for training and evaluating the machine
        learning model. It contains information about training and validation datasets,
        model parameters, and evaluation methods.
    :type trainer: Any

    :ivar params: A dictionary where keys correspond to parameter names and values
        are the lists of hyperparameter values to explore during tuning.
    :type params: dict

    :ivar results: A DataFrame containing the aggregated performance metrics
        for each evaluated hyperparameter combination.
    :type results: pd.DataFrame

    :ivar train_data: A DataFrame holding the training dataset.
    :type train_data: pd.DataFrame

    :ivar val_data: A DataFrame holding the validation data.
    :type val_data: pd.DataFrame
    """
    def __init__(self, trainer, params):
        self.trainer = trainer
        self.params = params
        self.results = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()

    @property
    def rnd_seed(self):
        """
        Gets the random seed value used by the trainer.

        :rtype: int
        :return: The random seed value.
        """
        return self.trainer.rnd_seed

    def get_combinations(self):
        """
        Generates all possible combinations of parameter values provided in a dictionary.

        This method utilizes itertools.product to create a cartesian product of all parameter
        values stored in the dictionary. Each parameter's values are iterated, producing all
        combinations of values across the provided parameters.

        :returns: An iterator over all possible combinations of the parameter values.
        :rtype: Iterator[Tuple[Any, ...]]
        """
        return it.product(*self.params.values())

    def get_best_params(self, by='R2', asc=False):
        """
        Sorts the results DataFrame to find the best set of parameters based on a
        specified metric and returns them as a dictionary. The sorting operation
        can be customized by specifying the column to sort by and the sort order.

        :param by: The column name in the results DataFrame used for sorting.
        :param asc: A boolean indicating whether to sort in ascending order.
        :return: A dictionary containing the best set of parameters based on the
            specified column and sorting order.
        """
        return self.results.sort_values(
            by=by,
            ascending=asc).iloc[0][list(self.params)].to_dict()

    def set_train_val_data(self):
        """
        The `train_data` attribute is updated with this new DataFrame,
        and the same happens for `val_data`. The current instance is returned.

        :raises AttributeError: If `trainer` does not have `train_df` or
            `val_df` attributes.

        :return: Returns the current instance with the `train_data` attribute
            updated.
        :rtype: object
        """
        self.train_data = self.trainer.train_df
        self.val_data = self.trainer.val_df
        return self

    def get_model_generator(self):
        combinations = self.get_combinations()
        for comb in combinations:
            trainer = self.trainer.__class__(
                self.trainer._train_file,
                self.trainer._test_file,
                self.trainer.model_file
            )
            yield trainer, comb, comb, list(self.params.keys()), self.train_data, self.val_data,

    # @profile
    def tune(self):
        """
        Tunes a machine learning model using cross-validation with all combinations of
        specified parameter values. Iterates over parameter combinations and evaluates
        model performance across folds, aggregating results.

        The method starts by generating all combinations of provided parameters and
        sets up k-fold cross-validation. For each combination and fold, it splits the
        training data, updates the model's parameters, and evaluates the model performance.
        The fold results are averaged and appended as part of the final results dataframe.
        This workflow ensures optimal parameter selection based on cross-validation
        performance.

        :param self: The instance of the class that executes the tuning process.
        :return: The updated instance of the class with tuning results stored.
        :rtype: object

        :raises Exception: Any exceptions encountered during model training or evaluation.
        """
        n_total = len(list(self.get_combinations()))
        par_name  = list(self.params.keys())
        # k_folds = KFold(n_splits=self.cv, shuffle=True, random_state=self.rnd_seed)
        pool = False

        if pool:
            with closing(Pool(8)) as pool:
                # fun = partial(
                #     HypeTuner.fit_model,
                #     par_name=par_name,
                #     train_data=self.train_data,
                #     val_data=self.val_data,
                # )
                results = pool.starmap(
                    self.fit_model,
                    self.get_model_generator()
                )
            self.results = pd.concat(results, ignore_index=True)
        else:
            combinations = self.get_combinations()
            for n, comb in enumerate(combinations):
                print(f"Progress: {n+1}/{n_total}")
                mean_perf = self.fit_model(self.trainer, comb, par_name, self.train_data, self.val_data)
                # Aggregate the mean values to results
                self.results = pd.concat([
                    self.results,
                    mean_perf
                ], ignore_index=True)
        return self

    @staticmethod
    def fit_model(trainer, comb, par_name, train_data, val_data):
        # Get the train and validation df
        trainer.train_df = train_data
        trainer.val_df = val_data
        # Update model parameters
        for k, v in zip(par_name, comb):
            trainer.ai_model.model_params[k] = v
        # Train the model
        trainer.train_model()
        # Get model metrics df
        perf = trainer.evaluate_model(
            x=trainer.val_df[trainer.x_colnames],
            y=trainer.val_df[trainer.y_colname]
        )
        del trainer.val_df, trainer.train_df

        for k, v in zip(par_name, comb): perf[k] = [v, ]
        return perf

    def pipeline(self):
        """
        Represents a sequence of actions or methods to process and execute a pipeline. The pipeline
        is designed to perform a series of connected operations in succession, where the output of
        one operation serves as the input for the next.

        The pipeline begins by setting up training data and proceeds to the tuning phase to finalize
        the process.

        :return: None
        """
        self.set_train_val_data().tune()
