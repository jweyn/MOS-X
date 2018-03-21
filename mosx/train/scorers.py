#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Scoring techniques for model evaluation.
"""

import numpy as np
from sklearn.metrics import make_scorer


def wxchallenge_error(y, y_pred, average=True, no_rain=False):
    """
    Returns the forecast error as measured by the WxChallenge.

    :param y: n-by-4 array of truth values
    :param y_pred: n-by-4 array of predicted values
    :param average: bool: if True, returns the average error per sample
    :param no_rain: bool: if True, does not count rain error
    :return: float: cumulative error
    """
    if y.shape != y_pred.shape:
        raise ValueError("y and y_pred must have the same shape")
    if len(y.shape) > 2:
        raise ValueError("got too many dimensions for y and y_pred (expected 2)")
    if (y.shape != (4,)) and y.shape[1] != 4:
        raise ValueError('last dimension of y and y_pred should be length 4')

    if len(y.shape) == 1:
        y = np.array([y])
        y_pred = np.array([y_pred])

    high_error = np.sum(np.abs(y[:, 0] - y_pred[:, 0]))
    low_error = np.sum(np.abs(y[:, 1] - y_pred[:, 1]))
    wind_error = 0.5 * np.sum(np.abs(y[:, 2] - y_pred[:, 2]))
    rain_error = 0.
    for sample in range(y.shape[0]):
        y_rain = y[sample, 3]
        y_pred_rain = y_pred[sample, 3]
        rain_min = int(100.*min(y_rain, y_pred_rain))
        rain_max = int(100.*max(y_rain, y_pred_rain))
        while rain_min < rain_max:
            if rain_min < 10:
                rain_error += 0.4
            elif rain_min < 25:
                rain_error += 0.3
            elif rain_min < 50:
                rain_error += 0.2
            else:
                rain_error += 0.1
            rain_min += 1

    result = high_error + low_error + wind_error + rain_error
    if average:
        result /= y.shape[0]
    return result


def wxchallenge_scorer(**kwargs):
    """
    Return a scikit-learn scorer object for forecast error as measured by WxChallenge.

    :param kwargs: parameters passed to the WxChallenge error function
        no_rain: if True, does not count rain error
    :return:
    """
    scorer = make_scorer(wxchallenge_error, greater_is_better=False, **kwargs)
    return scorer
