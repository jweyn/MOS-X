#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Special classes containing MOS-X-specific scikit-learn estimators. These estimators are designed to perform specialized
functions such as adding time series prediction, creating sub-estimators to tune the rain prediction, and so on. They
are NOT fully-compatible with scikit-learn because it is impossible to link all of the sklearn attributes of the base
estimators to these meta-estimators. It might be possible to, in the future, make these meta-classes that inherit from
the base sklearn estimator class.
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from copy import deepcopy
from .util import get_object


class TimeSeriesEstimator(object):
    """
    Wrapper class for containing separately-trained daily and timeseries estimators.
    """
    def __init__(self, daily_estimator, timeseries_estimator):
        self.daily_estimator = daily_estimator
        self.timeseries_estimator = timeseries_estimator
        # Inherit attributes from the daily estimator by default.
        # Apparently only 'steps' and 'memory' are in __dict__ for a Pipeline. BS.
        for attr in self.daily_estimator.__dict__.keys():
            try:
                setattr(self, attr, getattr(self.daily_estimator, attr))
            except AttributeError:
                pass
        # Apparently still have to do this
        if isinstance(self.daily_estimator, Pipeline):
            self.named_steps = self.daily_estimator.named_steps
        else:
            self.named_steps = None
        try:  # Likely only works if model has been fitted
            self.estimators_ = self.daily_estimator.estimators_
        except AttributeError:
            self.estimators_ = None
        self.array_form = True
        if not hasattr(self, 'verbose'):
            self.verbose = 1

    def fit(self, predictor_array, verification_array, **kwargs):
        """
        Fit both the daily and the timeseries estimators.

        :param predictor_array: ndarray-like: predictor features
        :param verification_array: ndarray-like: truth values
        :param kwargs: kwargs passed to fit methods
        :return:
        """
        if self.verbose > 0:
            print('TimeSeriesEstimator: fitting DAILY estimator')
        self.daily_estimator.fit(predictor_array, verification_array[:, :4], **kwargs)
        if self.verbose > 0:
            print('TimeSeriesEstimator: fitting TIMESERIES estimator')
        self.timeseries_estimator.fit(predictor_array, verification_array[:, 4:], **kwargs)

    def predict(self, predictor_array, **kwargs):
        """
        Predict from both the daily and timeseries estimators. Returns an array if self.array_form is True,
        otherwise returns a dictionary (not implemented yet).

        :param predictor_array: num_samples x num_features
        :param kwargs: kwargs passed to predict methods
        :return: array or dictionary of predicted values
        """
        daily = self.daily_estimator.predict(predictor_array, **kwargs)
        timeseries = self.timeseries_estimator.predict(predictor_array, **kwargs)
        if self.array_form:
            return np.concatenate((daily, timeseries), axis=1)


class RainTuningEstimator(object):
    """
    This class extends an estimator to include a separately-trained post-processing random forest for the daily rainfall
    prediction. Standard algorithms generally do a poor job of predicting a variable that has such a non-normal
    probability distribution as daily rainfall (which is dominated by 0s).
    """
    def __init__(self, estimator, rain_estimator='ensemble.RandomForestRegressor', **kwargs):
        """
        Initialize an instance of an estimator with a rainfall post-processor.

        :param estimator: sklearn estimator or TimeSeriesEstimator with an estimators_ attribute
        :param rain_estimator: str: type of sklearn estimator to use for rain processing
        :param kwargs: passed to sklearn rain estimator
        """
        self.base_estimator = estimator
        if isinstance(estimator, TimeSeriesEstimator):
            self.daily_estimator = self.base_estimator.daily_estimator
        else:
            self.daily_estimator = self.base_estimator
        # Inherit attributes from the base estimator
        for attr in self.base_estimator.__dict__.keys():
            try:
                setattr(self, attr, getattr(self.base_estimator, attr))
            except AttributeError:
                pass
        try:
            self.estimators_ = self.base_estimator.estimators_
        except AttributeError:
            pass
        self.rain_estimator_name = rain_estimator
        self.is_classifier = ('Classifi' in self.rain_estimator_name)
        Regressor = get_object('sklearn.%s' % rain_estimator)
        self.rain_estimator = Regressor(**kwargs)
        if isinstance(self.daily_estimator, Pipeline):
            self.named_steps = self.daily_estimator.named_steps
            self._forest = self.daily_estimator.named_steps['regressor']
            self._imputer = self.daily_estimator.named_steps['imputer']
        else:
            self.named_steps = None
            self._imputer = None
            self._forest = self.daily_estimator
        if not hasattr(self, 'verbose'):
            self.verbose = 1

    def _get_tree_rain_prediction(self, X):
        # Get predictions from individual trees.
        num_samples = X.shape[0]
        if self._imputer is not None:
            X = self._imputer.transform(X)
        if isinstance(self._forest, MultiOutputRegressor):
            num_trees = len(self._forest.estimators_[0].estimators_)
            predicted_rain = np.zeros((num_samples, num_trees))
            for s in range(num_samples):
                Xs = X[s].reshape(1, -1)
                for t in range(num_trees):
                    try:
                        predicted_rain[s, t] = self._forest.estimators_[3].estimators_[t].predict(Xs)
                    except AttributeError:
                        # Work around the 2-D array of estimators for GBTrees
                        predicted_rain[s, t] = self._forest.estimators_[3].estimators_[t][0].predict(Xs)
        else:
            num_trees = len(self._forest.estimators_)
            predicted_rain = np.zeros((num_samples, num_trees))
            for s in range(num_samples):
                Xs = X[s].reshape(1, -1)
                for t in range(num_trees):
                    try:
                        predicted_rain[s, t] = self._forest.estimators_[t].predict(Xs)
                    except AttributeError:
                        # Work around an error in sklearn where GBTrees have length-1 ndarrays...
                        predicted_rain[s, t] = self._forest.estimators_[t][0].predict(Xs)
        return predicted_rain

    def _get_distribution(self, p_rain):
        # Get the mean, std, and number of 0 forecasts from the estimator.
        mean = np.mean(p_rain, axis=1)
        std = np.std(p_rain, axis=1)
        zero_frac = 1. * np.sum(p_rain < 0.01, axis=1) / p_rain.shape[1]
        return np.stack((mean, std, zero_frac), axis=1)

    def fit(self, predictor_array, verification_array, rain_array=None, **kwargs):
        """
        Fit the estimator and the post-processor.

        :param predictor_array: ndarray-like: predictor features
        :param verification_array: ndarray-like: truth values
        :param rain_array: ndarray-like: raw rain from the models
        :param kwargs: passed to the estimator's 'fit' method
        :return:
        """
        # First, fit the estimator as usual
        self.base_estimator.fit(predictor_array, verification_array, **kwargs)

        # Now generate the distribution information from the individual trees in the forest
        if self.verbose > 0:
            print('RainTuningEstimator: getting ensemble rain predictions')
        predicted_rain = self._get_tree_rain_prediction(predictor_array)
        rain_distribution = self._get_distribution(predicted_rain)
        # If raw rain values are given, add those to the distribution
        if rain_array is not None:
            rain_distribution = np.concatenate((rain_distribution, rain_array), axis=1)

        # Fit the rain estimator
        if self.verbose > 0:
            print('RainTuningEstimator: fitting rain post-processor')
        # # If we're using a classifier, then we may need to binarize the labels
        # if self.is_classifier:
        #     lb = LabelBinarizer()
        #     rain_targets = lb.fit_transform(verification_array[:, 3])
        # else:
        rain_targets = verification_array[:, 3]
        self.rain_estimator.fit(rain_distribution, rain_targets)

    def predict(self, predictor_array, rain_tuning=True, rain_array=None, **kwargs):
        """
        Return a prediction from the estimator with post-processed rain.

        :param predictor_array: ndarray-like: predictor features
        :param rain_tuning: bool: toggle option to disable rain tuning in prediction
        :param rain_array: ndarray-like: raw rain values from models. Must be provided if fit() was called using raw
        rain values and rain_tuning is True.
        :param kwargs: passed to estimator's 'predict' method
        :return: array of predictions
        """
        # Predict with the estimator as usual
        predicted = self.base_estimator.predict(predictor_array, **kwargs)

        # Now get the tuned rain
        if rain_tuning:
            if self.verbose > 0:
                print('RainTuningEstimator: tuning rain prediction')
            # Get the distribution from individual trees
            predicted_rain = self._get_tree_rain_prediction(predictor_array)
            rain_distribution = self._get_distribution(predicted_rain)
            if rain_array is not None:
                rain_distribution = np.concatenate((rain_distribution, rain_array), axis=1)
            tuned_rain = self.rain_estimator.predict(rain_distribution)
            predicted[:, 3] = tuned_rain

        return predicted

    def predict_proba(self, predictor_array, rain_tuning=True, rain_array=None, **kwargs):
        """
        Return a prediction from the estimator with post-processed rain, with a probability of rainfall. Should only
        be used if rain_forecast_type is 'pop'.

        :param predictor_array: ndarray-like: predictor features
        :param rain_tuning: bool: toggle option to disable rain tuning in prediction
        :param rain_array: ndarray-like: raw rain values from models. Must be provided if fit() was called using raw
        rain values and rain_tuning is True.
        :param kwargs: passed to estimator's 'predict' method
        :return: array of predictions
        """
        # Predict with the estimator as usual
        predicted = self.base_estimator.predict(predictor_array, **kwargs)

        # Do the probabilistic prediction for rain
        if rain_tuning:
            if self.verbose > 0:
                print('RainTuningEstimator: tuning rain prediction')
            # Get the distribution from individual trees
            predicted_rain = self._get_tree_rain_prediction(predictor_array)
            rain_distribution = self._get_distribution(predicted_rain)
            if rain_array is not None:
                rain_distribution = np.concatenate((rain_distribution, rain_array), axis=1)
            tuned_rain = self.rain_estimator.predict_proba(rain_distribution)
            predicted[:, 3] = np.sum(tuned_rain[:, 1:], axis=1)

        return predicted

    def predict_rain_proba(self, predictor_array, rain_array=None):
        """
        Get the raw categorical probabilistic prediction from the rain post-processor.
        :param predictor_array: ndarray-like: predictor features
        :param rain_array: ndarray-like: raw rain values from models. Must be provided if fit() was called using raw
        rain values.
        :return: array of categorical rain predictions
        """
        if self.verbose > 0:
            print('RainTuningEstimator: getting probabilistic rain prediction')
        # Get the distribution from individual trees
        predicted_rain = self._get_tree_rain_prediction(predictor_array)
        rain_distribution = self._get_distribution(predicted_rain)
        if rain_array is not None:
            rain_distribution = np.concatenate((rain_distribution, rain_array), axis=1)
        categorical_rain = self.rain_estimator.predict_proba(rain_distribution)
        return categorical_rain


class BootStrapEnsembleEstimator(object):
    """
    This class implements a bootstrapping technique to generate a small ensemble of identical algorithms trained on
    a random subset of the training data. Options include partitioning the training data evenly, so that no algorithm
    has any sample in any other algorithm's training set, or completely randomly, so that there may be an arbitrary
    amount of overlap in the individual algorithms' training sets.
    """
    def __init__(self, estimator, n_members=10, n_samples_split=0.1, unique_splits=False):
        """
        Initialize instance of a BootStrapEnsembleEstimator.
        :param estimator: object: base estimator object (scikit-learn or one of those in this module)
        :param n_members: int: number of bootstrapped ensemble members (individual trained models)
        :param n_samples_split: float or int: if float, then gives the fraction of the training set that is used to
        form the training set for any given ensemble member. If int, then gives the exact number of samples used.
        Ignored if unique_splits is True, because then each training set has the maximum number of samples possible to
        be unique.
        :param unique_splits: bool: if True, each split contains a unique set of samples not present in any other split
        """
        self.base_estimator = estimator
        self.n_members = n_members
        self.estimators_ = np.empty((n_members,), dtype=np.object)
        for n in range(n_members):
            self.estimators_[n] = deepcopy(estimator)
        self.n_samples_split = n_samples_split
        self.unique_splits = unique_splits
        if not hasattr(self.base_estimator, 'verbose'):
            self.verbose = 1
        else:
            self.verbose = self.base_estimator.verbose

    def get_splits(self, X, y):
        X_splits = []
        y_splits = []
        if self.unique_splits:
            kf = KFold(n_splits=self.n_members, shuffle=True)
            for train_index, test_index in kf.split(X):
                X_splits.append(X[test_index])  # test is unique; train is all other tests
                y_splits.append(y[test_index])
        else:
            for split_count in range(self.n_members):
                X_split, xt, y_split, yt = train_test_split(X, y, train_size=self.n_samples_split, shuffle=True)
                X_splits.append(X_split)
                y_splits.append(y_split)
        return X_splits, y_splits

    def fit(self, predictor_array, verification_array, **kwargs):
        """
        Fit the bootstrapped algorithms.

        :param predictor_array: ndarray-like: predictor features
        :param verification_array: ndarray-like: truth values
        :param kwargs: kwargs passed to fit methods
        :return:
        """
        predictor_splits, verification_splits = self.get_splits(predictor_array, verification_array)
        for est in range(self.n_members):
            if self.verbose:
                print('BootStrapEnsembleEstimator: training ensemble member %d of %d' % (est + 1, self.n_members))
            self.estimators_[est].fit(predictor_splits[est], verification_splits[est], **kwargs)

    def predict(self, predictor_array, **kwargs):
        """
        Predict from the bootstrapped algorithms. Gives an ensemble mean.

        :param predictor_array: ndarray-like: predictor features
        :param kwargs: passed to estimator's 'predict' method
        :return:
        """
        prediction = []
        for est in range(self.n_members):
            if self.verbose:
                print('BootStrapEnsembleEstimator: predicting from ensemble member %d of %d' %
                      (est + 1, self.n_members))
            prediction.append(self.estimators_[est].predict(predictor_array, **kwargs))

        return np.mean(np.array(prediction), axis=0)

    def predict_rain_proba(self, predictor_array, rain_array=None):
        """
        If the base estimator is a RainTuningEstimator, yields an average prediction for categorical rain
        probabilities.
        :param predictor_array: ndarray-like: predictor features
        :param rain_array: ndarray-like: raw rain values from models. Must be provided if fit() was called using raw
        rain values.
        :return:
        """
        prediction = []
        for est in range(self.n_members):
            try:
                prediction.append(self.estimators_[est].predict_rain_proba(predictor_array, rain_array=rain_array))
            except AttributeError:
                raise AttributeError("'%s' cannot predict rain category probabilities; use RainTuningEstimator" %
                                     type(self.base_estimator))

        # Fix the shape of the arrays, in case some predictions don't have all categories of rain
        rain_dims = [p.shape[1] for p in prediction]
        max_dim = np.max(rain_dims)
        new_prediction = []
        for p in prediction:
            while p.shape[1] < max_dim:
                p = np.c_[p, np.zeros(p.shape[0])]
            new_prediction.append(p)

        return np.mean(np.array(new_prediction), axis=0)
