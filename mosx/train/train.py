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


def build_estimator(config):
    """
    Build the estimator object from the parameters in config.

    :param config:
    :return:
    """
    regressor = config['Model']['regressor']
    sklearn_kwargs = config['Model']['kwargs']
    train_individual = config['Model']['train_individual']
    ada_boost = config['Model'].get('ada_boost', None)
    Regressor = get_object('sklearn.%s' % regressor)
    if config['verbose']:
        print('Using sklearn.%s as estimator...' % regressor)

    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler as Scaler
    from sklearn.pipeline import Pipeline

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
        Scoring function for the error calculation; should be a sciki-learn scorer object

    title : string
        Title for the chart.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes : iterable, optional
        Sequence of subsets of training data used in learning curve plot
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    estimator = build_estimator(config)
    X, y = build_train_data(config, predictor_file, no_obs=no_obs, no_models=no_models)

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
