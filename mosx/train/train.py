#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for training scikit-learn models.
"""

from mosx.util import get_object, TimeSeriesEstimator
import pickle
import numpy as np


def train(config, predictor_file, output_file=None, no_obs=False, no_models=False, test_size=0,
          regressor='ensemble.RandomForestRegressor', sklearn_kwargs={}, train_individual=False, ada_boost=None):
    """
    Generate and train a scikit-learn machine learning estimator. The estimator object is saved as a pickle so that it
    may be imported and used for predictions at any time.

    :param config:
    :param predictor_file: str: full path to saved file of predictor data
    :param output_file: str: full path to output model file
    :param no_obs: bool: if True, generates the model with no OBS data
    :param no_models: bool: if True, generates the model with no BUFR data
    :param test_size: int: if > 0, returns a subset of the training data of size 'test_size' to test on
    :param regressor: str: name of the scikit-learn base regressor object
    :param sklearn_kwargs: dict: keyword args passed to the scikit-learn regressor
    :param train_individual: bool: if True, trains a separate model for each output variable
    :param ada_boost: dict: if not None, uses an Ada boosting meta-estimator with kwargs specified here
    :return:
    """
    Regressor = get_object('sklearn.%s' % regressor)
    if config['verbose']:
        print('Using sklearn.%s as estimator...' % regressor)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler as Scaler
    from sklearn.pipeline import Pipeline

    if config['verbose']:
        print('Reading predictor file...')
    with open(predictor_file, 'rb') as handle:
        data = pickle.load(handle)

    # Select data; impute missing values in BUFKIT and OBS
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

    num_samples = targets.shape[0]
    if test_size > 0:
        p_train, p_test, t_train, t_test = train_test_split(predictors, targets,
                                                            train_size=num_samples - test_size)
    else:
        p_train = predictors
        t_train = targets

    # Create and train the learning algorithm
    # Set default kwargs for neural net, random forest
    if regressor.startswith('neural_net'):
        if 'activation' not in sklearn_kwargs:
            sklearn_kwargs['activation'] = 'logistic'
    elif regressor.startswith('ensemble'):
        if 'n_estimators' not in sklearn_kwargs:
            sklearn_kwargs['n_estimators'] = 250

    print('Here are the parameters passed to the learning algorithm...')
    print(sklearn_kwargs)

    # Create the pipeline list
    pipeline = [("imputer", Imputer(missing_values=np.nan, strategy="mean", axis=0))]
    if config['Model']['predict_timeseries']:
        pipeline_timeseries = [("imputer", Imputer(missing_values=np.nan, strategy="mean", axis=0))]

    if not(regressor.startswith('ensemble')):
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

    print('Training the algorithm...')
    estimator.fit(p_train, t_train)

    if output_file is None:
        output_file = '%s/%s_mosx.pkl' % (config['MOSX_ROOT'], config['station_id'])
    print('-> Exporting to %s' % output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if test_size > 0:
        return p_test, t_test
    return
