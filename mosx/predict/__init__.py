#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for predicting from scikit-learn models.
"""

import pickle
import numpy as np
import pandas as pd
from ..util import dewpoint, to_bool


def predict(config, predictor_file, ensemble=False, time_series_date=None, tune_rain=False):
    """
    Predict forecasts from the estimator in config.

    :param config:
    :param predictor_file: str: file containing predictor data from mosx.train.format_predictors
    :param ensemble: bool: if True, return an array of num_trees-by-4 of the predictions of each tree in the estimator
    :param time_series_date: datetime: if set, returns a time series prediction from the estimator, where the datetime
    provided is the day the forecast is for (only works for single-day runs, or assumes last day)
    :param tune_rain: bool: if True, applies manual tuning to the rain forecast
    :return:
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
    if rain_tuning is not None and to_bool(rain_tuning.get('use_raw_rain', False)):
        predicted = estimator.predict(predictors, rain_array=predictor_data.rain)
    else:
        predicted = estimator.predict(predictors)
    precip = predictor_data.rain

    # Check for precipitation override
    if tune_rain:
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

    # Round off daily values
    predicted[:, :3] = np.round(predicted[:, :3])
    predicted[:, 3] = np.round(predicted[:, 3], 2)

    # If probabilities are requested and available, get the results from each tree
    if not (config['Model']['regressor'].startswith('ensemble')):
        ensemble = False
    if ensemble:
        imputer = estimator.named_steps['imputer']
        forest = estimator.named_steps['regressor']
        predictors = imputer.transform(predictors)
        if config['Model']['train_individual']:
            num_trees = len(forest.estimators_[0].estimators_)
            all_predicted = np.zeros((num_trees, 4))
            for v in range(4):
                for t in range(num_trees):
                    try:
                        all_predicted[t, v] = forest.estimators_[v].estimators_[t].predict(predictors)
                    except AttributeError:
                        # Work around the 2-D array of estimators for GBTrees
                        all_predicted[t, v] = forest.estimators_[v].estimators_[t][0].predict(predictors)
        else:
            num_trees = len(forest.estimators_)
            all_predicted = np.zeros((num_trees, 4))
            for t in range(num_trees):
                try:
                    all_predicted[t, :] = forest.estimators_[t].predict(predictors)[:4]
                except AttributeError:
                    # Work around the 2-D array of estimators for GBTrees
                    all_predicted[t, :] = forest.estimators_[t][0].predict(predictors)[:4]
    else:
        all_predicted = None

    if config['Model']['predict_timeseries'] and time_series_date is not None:
        predicted_array = predicted[-1, 4:].reshape((4, 25)).T
        # Get dewpoint
        predicted_array[:, 2] = dewpoint(predicted_array[:, 0], predicted_array[:, 2])
        times = pd.date_range(time_series_date.replace(hour=6), periods=25, freq='H').to_pydatetime().tolist()
        variables = ['temperature', 'rain', 'dewpoint', 'windSpeed']
        round_dict = {'temperature': 0, 'rain': 2, 'dewpoint': 0, 'windSpeed': 0}
        predicted_timeseries = pd.DataFrame(predicted_array, index=times, columns=variables)
        predicted_timeseries = predicted_timeseries.round(round_dict)
    else:
        predicted_timeseries = None

    return predicted, all_predicted, predicted_timeseries
