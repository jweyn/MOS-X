#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for training scikit-learn models.
"""

from datetime import datetime, timedelta
import pandas as pd

from mosx.util import get_object, to_bool, dewpoint
from mosx.estimators import TimeSeriesEstimator, RainTuningEstimator, BootStrapEnsembleEstimator
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
    bootstrap = config['Model'].get('Bootstrapping', None)
    Regressor = get_object('sklearn.%s' % regressor)
    if config['verbose']:
        print('build_estimator: using sklearn.%s as estimator...' % regressor)

    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler as Scaler
    from sklearn.pipeline import Pipeline

    # Create and train the learning algorithm
    if config['verbose']:
        print('build_estimator: here are the parameters passed to the learning algorithm...')
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
        if config['verbose']:
            print('build_estimator: using Ada boosting...')
        from sklearn.ensemble import AdaBoostRegressor
        regressor_obj = AdaBoostRegressor(regressor_obj, **ada_boost)
    if train_individual:
        if config['verbose']:
            print('build_estimator: training separate models for each parameter...')
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
        if config['verbose']:
            print('build_estimator: using rain tuning...')
        rain_kwargs = rain_tuning.copy()
        rain_kwargs.pop('use_raw_rain', None)
        estimator = RainTuningEstimator(estimator, **rain_kwargs)

    # Add bootstrapping if requested
    if bootstrap is not None:
        if config['verbose']:
            print('build_estimator: using bootstrapping ensemble...')
        estimator = BootStrapEnsembleEstimator(estimator, **bootstrap)

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
        print('build_train_data: reading predictor file')
    rain_tuning = config['Model'].get('Rain tuning', None)
    with open(predictor_file, 'rb') as handle:
        data = pickle.load(handle)

    # Select data
    if no_obs and no_models:
        no_obs = False
        no_models = False
    if no_obs:
        if config['verbose']:
            print('build_train_data: not using observations to train')
        predictors = data['BUFKIT']
    elif no_models:
        if config['verbose']:
            print('build_train_data: not using models to train')
        predictors = data['OBS']
    else:
        predictors = np.concatenate((data['BUFKIT'], data['OBS']), axis=1)
    if rain_tuning is not None and to_bool(rain_tuning.get('use_raw_rain', False)):
        predictors = np.concatenate((predictors, data.rain), axis=1)
        rain_shape = data.rain.shape[-1]
    targets = data['VERIF']

    if test_size > 0:
        p_train, p_test, t_train, t_test = train_test_split(predictors, targets, test_size=test_size)
        if rain_tuning is not None and to_bool(rain_tuning.get('use_raw_rain', False)):
            r_train = p_train[:, -1*rain_shape:]
            p_train = p_train[:, :-1*rain_shape]
            r_test = p_test[:, -1 * rain_shape:]
            p_test = p_test[:, :-1 * rain_shape]
        else:
            r_train = None
            r_test = None
        return p_train, t_train, r_train, p_test, t_test, r_test
    else:
        if rain_tuning is not None and to_bool(rain_tuning.get('use_raw_rain', False)):
            return predictors, targets, data.rain
        else:
            return predictors, targets, None


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
    rain_tuning = config['Model'].get('Rain tuning', None)
    if test_size > 0:
        p_train, t_train, r_train, p_test, t_test, r_test = build_train_data(config, predictor_file, no_obs=no_obs,
                                                                             no_models=no_models, test_size=test_size)
    else:
        p_train, t_train, r_train = build_train_data(config, predictor_file, no_obs=no_obs, no_models=no_models)

    print('train: training the estimator')
    if rain_tuning is not None and to_bool(rain_tuning.get('use_raw_rain', False)):
        estimator.fit(p_train, t_train, rain_array=r_train)
    else:
        estimator.fit(p_train, t_train)

    if estimator_file is None:
        estimator_file = '%s/%s_mosx.pkl' % (config['MOSX_ROOT'], config['station_id'])
    print('train: -> exporting to %s' % estimator_file)
    with open(estimator_file, 'wb') as handle:
        pickle.dump(estimator, handle, protocol=2)

    if test_size > 0:
        return p_test, t_test, r_test
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
        Produces arrays of indices to use for model and test splits.

        :param X: array-like, shape (samples, features): predictor data
        :param y: array-like, shape (samples, outputs) or None: verification data; ignored
        :param groups: ignored
        :return: model, test: 1-D arrays of sample indices in the model and test sets
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
        Return the number of splits. Dummy function for compatibility.

        :param X: ignored
        :param y: ignored
        :param groups: ignored
        :return:
        """
        return self.n_splits


def predict_all(config, predictor_file, ensemble=False, time_series_date=None, naive_rain_correction=False,
                round_result=False, **kwargs):
    """
    Predict forecasts from the estimator in config. Also return probabilities and time series.

    :param config:
    :param predictor_file: str: file containing predictor data from mosx.model.format_predictors
    :param ensemble: bool: if True, return an array of num_trees-by-4 of the predictions of each tree in the estimator
    :param time_series_date: datetime: if set, returns a time series prediction from the estimator, where the datetime
    provided is the day the forecast is for (only works for single-day runs, or assumes last day)
    :param naive_rain_correction: bool: if True, applies manual tuning to the rain forecast
    :param round_result: bool: if True, rounds the predicted estimate
    :param kwargs: passed to the estimator's 'predict' method
    :return:
    predicted: ndarray: num_samples x num_predicted_variables predictions
    all_predicted: ndarray: num_samples x num_predicted_variables x num_ensemble_members predictions for all trees
    predicted_timeseries: DataFrame: time series for final sample
    """
    # Load the predictor data and estimator
    with open(predictor_file, 'rb') as handle:
        predictor_data = pickle.load(handle)
    rain_tuning = config['Model'].get('Rain tuning', None)
    if config['verbose']:
        print('predict: loading estimator %s' % config['Model']['estimator_file'])
    with open(config['Model']['estimator_file'], 'rb') as handle:
        estimator = pickle.load(handle)

    predictors = np.concatenate((predictor_data['BUFKIT'], predictor_data['OBS']), axis=1)
    if config['Model']['rain_forecast_type'] == 'pop' and getattr(estimator, 'is_classifier', False):
        predict_method = estimator.predict_proba
    else:
        predict_method = estimator.predict
    if rain_tuning is not None and to_bool(rain_tuning.get('use_raw_rain', False)):
        predicted = predict_method(predictors, rain_array=predictor_data.rain, **kwargs)
    else:
        predicted = predict_method(predictors, **kwargs)
    precip = predictor_data.rain

    # Check for precipitation override
    if naive_rain_correction:
        for day in range(predicted.shape[0]):
            if sum(precip[day]) < 0.01:
                if config['verbose']:
                    print('predict: warning: overriding MOS-X rain prediction of %0.2f on day %s with 0' %
                          (predicted[day, 3], day))
                predicted[day, 3] = 0.
            elif predicted[day, 3] > max(precip[day]) or predicted[day, 3] < min(precip[day]):
                if config['verbose']:
                    print('predict: warning: overriding MOS-X prediction of %0.2f on day %s with model mean' %
                          (predicted[day, 3], day))
                predicted[day, 3] = max(0., np.mean(precip[day] + [predicted[day, 3]]))
    else:
        # At least make sure we aren't predicting negative values...
        predicted[:, 3][predicted[:, 3] < 0] = 0.0

    # Round off daily values, if selected
    if round_result:
        predicted[:, :3] = np.round(predicted[:, :3])
        predicted[:, 3] = np.round(predicted[:, 3], 2)

    # If probabilities are requested and available, get the results from each tree
    if ensemble:
        num_samples = predictors.shape[0]
        if not hasattr(estimator, 'named_steps'):
            forest = estimator
        else:
            imputer = estimator.named_steps['imputer']
            forest = estimator.named_steps['regressor']
            predictors = imputer.transform(predictors)
        # If we generated our own ensemble by bootstrapping, it must be treated as such
        if config['Model']['train_individual'] and config['Model'].get('Bootstrapping', None) is None:
            num_trees = len(forest.estimators_[0].estimators_)
            all_predicted = np.zeros((num_samples, 4, num_trees))
            for v in range(4):
                for t in range(num_trees):
                    try:
                        all_predicted[:, v, t] = forest.estimators_[v].estimators_[t].predict(predictors)
                    except AttributeError:
                        # Work around the 2-D array of estimators for GBTrees
                        all_predicted[:, v, t] = forest.estimators_[v].estimators_[t][0].predict(predictors)
        else:
            num_trees = len(forest.estimators_)
            all_predicted = np.zeros((num_samples, 4, num_trees))
            for t in range(num_trees):
                try:
                    all_predicted[:, :, t] = forest.estimators_[t].predict(predictors)[:, :4]
                except AttributeError:
                    # Work around the 2-D array of estimators for GBTrees
                    all_predicted[:, :, t] = forest.estimators_[t][0].predict(predictors)[:, :4]
    else:
        all_predicted = None

    if config['Model']['predict_timeseries']:
        if time_series_date is None:
            date_now = datetime.utcnow()
            time_series_date = datetime(date_now.year, date_now.month, date_now.day) + timedelta(days=1)
            print('predict: warning: set time series start date to %s (was unspecified)' % time_series_date)
        num_hours = int(24 / config['time_series_interval']) + 1
        predicted_array = predicted[-1, 4:].reshape((4, num_hours)).T
        # Get dewpoint
        predicted_array[:, 2] = dewpoint(predicted_array[:, 0], predicted_array[:, 2])
        times = pd.date_range(time_series_date.replace(hour=6), periods=num_hours,
                              freq='%dH' % config['time_series_interval']).to_pydatetime().tolist()
        variables = ['temperature', 'rain', 'dewpoint', 'windSpeed']
        round_dict = {'temperature': 0, 'rain': 2, 'dewpoint': 0, 'windSpeed': 0}
        predicted_timeseries = pd.DataFrame(predicted_array, index=times, columns=variables)
        predicted_timeseries = predicted_timeseries.round(round_dict)
    else:
        predicted_timeseries = None

    return predicted, all_predicted, predicted_timeseries


def predict(config, predictor_file, naive_rain_correction=False, round=False, **kwargs):
    """
    Predict forecasts from the estimator in config. Only returns daily values.

    :param config:
    :param predictor_file: str: file containing predictor data from mosx.model.format_predictors
    :param naive_rain_correction: bool: if True, applies manual tuning to the rain forecast
    :param round: bool: if True, rounds the predicted estimate
    :param kwargs: passed to the estimator's 'predict' method
    :return:
    """

    predicted, all_predicted, predicted_timeseries = predict_all(config, predictor_file,
                                                                 naive_rain_correction=naive_rain_correction,
                                                                 round_result=round, **kwargs)
    return predicted


def predict_rain_proba(config, predictor_file):
    """
    Predict probabilistic rain forecasts for 'pop' or 'categorical' types.

    :param config:
    :param predictor_file: str: file containing predictor data from mosx.model.format_predictors
    :return:
    """
    if config['Model']['rain_forecast_type'] not in ['pop', 'categorical']:
        raise TypeError("'quantity' rain forecast is not probabilistic, cannot get probabilities")
    rain_tuning = config['Model'].get('Rain tuning', None)
    if rain_tuning is None:
        raise TypeError('Probabilistic rain forecasts are only possible with a RainTuningEstimator')

    # Load the predictor data and estimator
    with open(predictor_file, 'rb') as handle:
        predictor_data = pickle.load(handle)
    if config['verbose']:
        print('predict: loading estimator %s' % config['Model']['estimator_file'])
    with open(config['Model']['estimator_file'], 'rb') as handle:
        estimator = pickle.load(handle)

    predictors = np.concatenate((predictor_data['BUFKIT'], predictor_data['OBS']), axis=1)
    if to_bool(rain_tuning.get('use_raw_rain', False)):
        rain_proba = estimator.predict_rain_proba(predictors, rain_array=predictor_data.rain)
    else:
        rain_proba = estimator.predict_rain_proba(predictors)

    return rain_proba
