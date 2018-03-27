#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for training scikit-learn models.
"""

from mosx.util import get_object, TimeSeriesEstimator, RainTuningEstimator
import pickle
import numpy as np


def build_estimator(config):
    """
    Build the estimator object from the parameters in config.

    :param config:
    :return:
    """
    regressor = config['Model']['regressor']
    sklearn_kwargs = config['Model']['Parameters']
    train_individual = config['Model']['train_individual']
    ada_boost = config['Model'].get('Ada boosting', None)
    rain_tuning = config['Model'].get('Rain tuning', None)
    Regressor = get_object('sklearn.%s' % regressor)
    if config['verbose']:
        print('Using sklearn.%s as estimator...' % regressor)

    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler as Scaler
    from sklearn.pipeline import Pipeline

    # Create and train the learning algorithm
    print('Here are the parameters passed to the learning algorithm...')
    print(sklearn_kwargs)

    # Create the pipeline list
    pipeline = [("imputer", Imputer(missing_values=np.nan, strategy="mean", axis=0))]
    if config['Model']['predict_timeseries']:
        pipeline_timeseries = [("imputer", Imputer(missing_values=np.nan, strategy="mean", axis=0))]

    if not (regressor.startswith('ensemble')):
        # Need to add feature scaling
        pipeline.append(("scaler", Scaler()))
        if config['Model']['predict_timeseries']:
            pipeline_timeseries.append(("scaler", Scaler()))

    # Create the regressor object
    regressor_obj = Regressor(**sklearn_kwargs)
    if ada_boost is not None:
        print('Using Ada boosting...')
        from sklearn.ensemble import AdaBoostRegressor
        regressor_obj = AdaBoostRegressor(regressor_obj, **ada_boost)
    if train_individual:
        print('Training separate models for each parameter...')
        from sklearn.multioutput import MultiOutputRegressor
        multi_regressor = MultiOutputRegressor(regressor_obj, 4)
        pipeline.append(("regressor", multi_regressor))
    else:
        pipeline.append(("regressor", regressor_obj))
    if config['Model']['predict_timeseries']:
        pipeline_timeseries.append(("regressor", regressor_obj))

    # Make the final estimator with a Pipeline
    if config['Model']['predict_timeseries']:
        estimator = TimeSeriesEstimator(Pipeline(pipeline), Pipeline(pipeline_timeseries))
    else:
        estimator = Pipeline(pipeline)

    if rain_tuning is not None and regressor.startswith('ensemble'):
        print('Using rain tuning...')
        estimator = RainTuningEstimator(estimator, **rain_tuning)

    return estimator


def build_train_data(config, predictor_file, no_obs=False, no_models=False, test_size=0):
    """
    Build the array of training (and optionally testing) data.

    :param config:
    :param predictor_file:
    :param no_obs:
    :param no_models:
    :param test_size:
    :return:
    """
    from sklearn.model_selection import train_test_split

    if config['verbose']:
        print('Reading predictor file...')
    with open(predictor_file, 'rb') as handle:
        data = pickle.load(handle)

    # Select data
    if no_obs and no_models:
        no_obs = False
        no_models = False
    if no_obs:
        if config['verbose']:
            print('Not using observations to train')
        predictors = data['BUFKIT']
    elif no_models:
        if config['verbose']:
            print('Not using models to train')
        predictors = data['OBS']
    else:
        predictors = np.concatenate((data['BUFKIT'], data['OBS']), axis=1)
    targets = data['VERIF']

    if test_size > 0:
        p_train, p_test, t_train, t_test = train_test_split(predictors, targets, test_size=test_size)
        return p_train, t_train, p_test, t_test
    else:
        p_train = predictors
        t_train = targets
        return p_train, t_train


def train(config, predictor_file, estimator_file=None, no_obs=False, no_models=False, test_size=0):
    """
    Generate and train a scikit-learn machine learning estimator. The estimator object is saved as a pickle so that it
    may be imported and used for predictions at any time.

    :param config:
    :param predictor_file: str: full path to saved file of predictor data
    :param estimator_file: str: full path to output model file
    :param no_obs: bool: if True, generates the model with no OBS data
    :param no_models: bool: if True, generates the model with no BUFR data
    :param test_size: int: if > 0, returns a subset of the training data of size 'test_size' to test on
    :return: matplotlib Figure if plot_learning_curve is True
    """
    estimator = build_estimator(config)
    if test_size > 0:
        p_train, t_train, p_test, t_test = build_train_data(config, predictor_file, no_obs=no_obs, no_models=no_models,
                                                            test_size=test_size)
    else:
        p_train, t_train = build_train_data(config, predictor_file, no_obs=no_obs, no_models=no_models)

    print('Training the estimator...')
    estimator.fit(p_train, t_train)

    if estimator_file is None:
        estimator_file = '%s/%s_mosx.pkl' % (config['MOSX_ROOT'], config['station_id'])
    print('-> Exporting to %s' % estimator_file)
    with open(estimator_file, 'wb') as handle:
        pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if test_size > 0:
        return p_test, t_test
    return


def _plot_learning_curve(estimator, X, y, ylim=None, cv=None, scoring=None, title=None, n_jobs=1,
                         train_sizes=np.linspace(.1, 1.0, 5)):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    fig = plt.figure()
    if title is not None:
        plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return fig


def plot_learning_curve(config, predictor_file, no_obs=False, no_models=False, ylim=None, cv=None, scoring=None,
                        title=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve. From scikit-learn:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    config :

    predictor_file : string
        Full path to file containing predictor data

    no_obs : boolean
        Train model without observations

    no_models : boolean
        Train model without model data

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    scoring :
        Scoring function for the error calculation; should be a scikit-learn scorer object

    title : string
        Title for the chart.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes : iterable, optional
        Sequence of subsets of training data used in learning curve plot
    """
    estimator = build_estimator(config)
    X, y = build_train_data(config, predictor_file, no_obs=no_obs, no_models=no_models)
    fig = _plot_learning_curve(estimator, X, y, ylim=ylim, cv=cv, scoring=scoring, title=title, n_jobs=n_jobs,
                               train_sizes=train_sizes)
    return fig


def combine_train_test(config, train_file, test_file, no_obs=False, no_models=False, return_count_test=True):
    """
    Concatenates the arrays of predictors and verification values from the train file and the test file. Useful for
    implementing cross-validation using scikit-learn's methods and the SplitConsecutive class.

    :param config:
    :param train_file: str: full path to predictor file for training
    :param test_file: str: full path to predictor file for validation
    :param no_obs: bool: if True, generates the model with no OBS data
    :param no_models: bool: if True, generates the model with no BUFR data
    :param return_count_test: bool: if True, also returns the number of samples in the test set (see SplitConsecutive)
    :return: predictors, verifications: concatenated arrays of predictors and verification values; count: number of
    samples in the test set
    """
    p_train, t_train = build_train_data(config, train_file, no_obs=no_obs, no_models=no_models)
    p_test, t_test = build_train_data(config, test_file, no_obs=no_obs, no_models=no_models)
    p_combined = np.concatenate((p_train, p_test), axis=0)
    t_combined = np.concatenate((t_train, t_test), axis=0)
    if return_count_test:
        return p_combined, t_combined, t_test.shape[0]
    else:
        return p_combined, t_combined


class SplitConsecutive(object):
    """
    Implements a split method to subset a training set into train and test sets, using the first or last n samples in
    the set.
    """

    def __init__(self, first=False, n_samples=0.2):
        """
        Create an instance of SplitConsecutive.

        :param first: bool: if True, gets test data from the beginning of the data set; otherwise from the end
        :param n_samples: float or int: if float, subsets a fraction (0 to 1) of the data into the test set; if int,
        subsets a specific number of samples.
        """
        if type(first) is not bool:
            raise TypeError("'first' must be a boolean type.")
        try:
            n_samples = int(n_samples)
        except:
            pass
        if type(n_samples) is float and (n_samples <= 0. or n_samples >= 1.):
            raise ValueError("if float, 'n_samples' must be between 0 and 1.")
        if type(n_samples) is not float and type(n_samples) is not int:
            raise TypeError("'n_samples' must be float or int type.")
        self.first = first
        self.n_samples = n_samples
        self.n_splits = 1

    def split(self, X, y=None, groups=None):
        """
        Produces arrays of indices to use for train and test splits.

        :param X: array-like, shape (samples, features): predictor data
        :param y: array-like, shape (samples, outputs) or None: verification data; ignored
        :param groups: ignored
        :return: train, test: 1-D arrays of sample indices in the train and test sets
        """
        num_samples = X.shape[0]
        indices = np.arange(0, num_samples, 1, dtype=np.int32)
        if type(self.n_samples) is float:
            self.n_samples = int(np.round(num_samples * self.n_samples))
        if self.first:
            test = indices[:self.n_samples]
            train = indices[self.n_samples:]
        else:
            test = indices[-self.n_samples:]
            train = indices[:num_samples - self.n_samples]
        yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splits.
        :param X: ignored
        :param y: ignored
        :param groups: ignored
        :return:
        """
        return self.n_splits
