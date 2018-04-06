#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for the MOS-X model.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==================================================================================================================== #
# Classes
# ==================================================================================================================== #

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
        self.named_steps = self.daily_estimator.named_steps
        self.array_form = True
        if not hasattr(self, 'verbose'):
            self.verbose = 1

    def fit(self, predictor_array, verification_array, **kwargs):
        """
        Fit both the daily and the timeseries estimators.
        :param predictor_array: num_samples x num_features
        :param verification_array: num_samples x num_daily+num_ts
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
    def __init__(self, estimator, **kwargs):
        """
        Initialize an instance of an estimator with a rainfall post-processor.

        :param estimator: sklearn estimator or TimeSeriesEstimator with an _estimators attribute
        :param kwargs: passed to scikit-learn random forest rain processing algorithm
        """
        self.base_estimator = estimator
        # Inherit attributes from the base estimator
        for attr in self.base_estimator.__dict__.keys():
            try:
                setattr(self, attr, getattr(self.base_estimator, attr))
            except AttributeError:
                pass
        self.named_steps = self.base_estimator.named_steps
        self.rain_processor = RandomForestRegressor(**kwargs)
        if isinstance(self.base_estimator, Pipeline):
            self._forest = self.base_estimator.named_steps['regressor']
            self._imputer = self.base_estimator.named_steps['imputer']
        else:
            self._imputer = None
            self._forest = self.base_estimator
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

        # Fit a random forest post-processor
        if self.verbose > 0:
            print('RainTuningEstimator: fitting rain post-processor')
        self.rain_processor.fit(rain_distribution, verification_array[:, 3])

    def predict(self, predictor_array, rain_tuning=True, rain_array=None, **kwargs):
        """
        Return a prediction from the estimator with post-processed rain.
        :param predictor_array: ndarray-like: predictor features
        :param rain_tuning: bool: toggle option to disable rain tuning in prediction
        :param rain_array: ndarray-like: raw rain values from models. Must be provided if fit() was called using raw
        rain values too and rain_tuning is True.
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
            tuned_rain = self.rain_processor.predict(rain_distribution)
            predicted[:, 3] = tuned_rain

        return predicted


# ==================================================================================================================== #
# Config functions
# ==================================================================================================================== #

def walk_kwargs(section, key):
    value = section[key]
    try:
        section[key] = int(value)
    except (TypeError, ValueError):
        try:
            section[key] = float(value)
        except (TypeError, ValueError):
            pass


def get_config(config_path):
    """
    Retrieve the config object from config_path.

    :param config_path: str: full path to config file
    :return:
    """
    import configobj
    from validate import Validator

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_spec = '%s/configspec' % dir_path

    try:
        config = configobj.ConfigObj(config_path, configspec=config_spec, file_error=True)
    except IOError:
        try:
            config = configobj.ConfigObj(config_path+'.config', configspec=config_spec, file_error=True)
        except IOError:
            print('Error: unable to open configuration file %s' % config_path)
            raise
    except configobj.ConfigObjError as e:
        print('Error while parsing configuration file %s' % config_path)
        print("*** Reason: '%s'" % e)
        raise

    config.validate(Validator())

    # Make sure site_directory is there
    if config['SITE_ROOT'] == '':
        config['SITE_ROOT'] = '%(MOSX_ROOT)s/site_data'

    # Make sure BUFR parameters have defaults
    if config['BUFR']['bufr_station_id'] == '':
        config['BUFR']['bufr_station_id'] = '%(station_id)s'
    if config['BUFR']['bufr_data_dir'] == '':
        config['BUFR']['bufr_data_dir'] = '%(SITE_ROOT)s/bufkit'
    if config['BUFR']['bufrgruven'] == '':
        config['BUFR']['bufrgruven'] = '%(BUFR_ROOT)s/bufr_gruven.pl'

    # Make sure Obs parameters have defaults
    if config['Obs']['sounding_data_dir'] == '':
        config['Obs']['sounding_data_dir'] = '%(SITE_ROOT)s/soundings'

    # Add in a list for BUFR models
    config['BUFR']['bufr_models'] = []
    for model in config['BUFR']['models']:
        if model.upper() == 'GFS':
            config['BUFR']['bufr_models'].append(['gfs3', 'gfs'])
        else:
            config['BUFR']['bufr_models'].append(model.lower())

    # Convert kwargs, Rain tuning, and Ada boosting, if available, to int or float types
    config['Model']['Parameters'].walk(walk_kwargs)
    try:
        config['Model']['Ada boosting'].walk(walk_kwargs)
    except KeyError:
        pass
    try:
        config['Model']['Rain tuning'].walk(walk_kwargs)
    except KeyError:
        pass

    return config


# ==================================================================================================================== #
# Utility functions
# ==================================================================================================================== #

def get_object(module_class):
    """
    Given a string with a module class name, it imports and returns the class.
    This function (c) Tom Keffer, weeWX.
    """

    # Split the path into its parts
    parts = module_class.split('.')
    # Strip off the classname:
    module = '.'.join(parts[:-1])
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts[1:]:
            mod = getattr(mod, part)
    except AttributeError:
        # Can't find something. Give a more informative error message:
        raise AttributeError("Module '%s' has no attribute '%s' when searching for '%s'" %
                             (mod.__name__, part, module_class))
    return mod


def generate_dates(config, api=False, start_date=None, end_date=None, api_add_hour=0):
    """
    Returns all of the dates requested from the config. If api is True, then returns a list of (start_date, end_date)
    tuples split by year in strings formatted for the MesoWest API call. If api is False, then returns a list of all
    dates as datetime objects. start_date and end_date are available as options as certain calls require addition of
    some data for prior days.

    :param config:
    :param api: bool: if True, returns dates formatted for MesoWest API call
    :param start_date: str: starting date in config file format (YYYYMMDD)
    :param end_date: str: ending date in config file format (YYYYMMDD)
    :param api_add_hour: int: add this number of hours to the end of the call, useful for getting up to 6Z on last day
    :return:
    """
    if start_date is None:
        start_date = datetime.strptime(config['data_start_date'], '%Y%m%d')
    if end_date is None:
        end_date = datetime.strptime(config['data_end_date'], '%Y%m%d')
    start_dt = start_date
    end_dt = end_date
    if start_dt > end_dt:
        raise ValueError('Start date must be before end date; check MOSX_INFILE.')
    end_year = end_dt.year + 1
    time_added = timedelta(hours=api_add_hour)
    all_dates = []
    if config['is_season']:
        if start_dt > datetime(start_dt.year, end_dt.month, end_dt.day):
            # Season crosses new year
            end_year -= 1
        for year in range(start_dt.year, end_year):
            if start_dt > datetime(start_dt.year, end_dt.month, end_dt.day):
                # Season crosses new year
                year2 = year + 1
            else:
                year2 = year
            if api:
                year_start = datetime.strftime(datetime(year, start_dt.month, start_dt.day), '%Y%m%d0000')
                year_end = datetime.strftime(datetime(year2, end_dt.month, end_dt.day) + time_added, '%Y%m%d%H00')
                all_dates.append((year_start, year_end))
            else:
                year_dates = pd.date_range(datetime(year, start_dt.month, start_dt.day),
                                           datetime(year2, end_dt.month, end_dt.day), freq='D')
                for date in year_dates:
                    all_dates.append(date.to_pydatetime())

    else:
        if api:
            for year in range(start_dt.year, end_year):
                if year == start_dt.year:
                    year_start = datetime.strftime(datetime(year, start_dt.month, start_dt.day), '%Y%m%d0000')
                else:
                    year_start = datetime.strftime(datetime(year, 1, 1), '%Y%m%d0000')
                if year == end_dt.year:
                    year_end = datetime.strftime(datetime(year, end_dt.month, end_dt.day) + time_added, '%Y%m%d%H00')
                else:
                    year_end = datetime.strftime(datetime(year+1, 1, 1) + time_added, '%Y%m%d%H00')
                all_dates.append((year_start, year_end))
        else:
            pd_dates = pd.date_range(start_dt, end_dt, freq='D')
            for date in pd_dates:
                all_dates.append(date.to_pydatetime())
    return all_dates


def find_matching_dates(bufr, obs, verif, return_data=False):
    """
    Finds dates which match in all three dictionaries. If return_data is True, returns the input dictionaries with only
    common dates retained. verif may be None if running the model.

    :param bufr: dict: dictionary of processed BUFR data
    :param obs: dict: dictionary of processed OBS data
    :param verif: dict: dictionary of processed VERIFICATION data
    :param return_data: bool: if True, returns edited data dictionaries containing only matching dates' data
    :return: list of dates[, new BUFR, OBS, and VERIF dictionaries]
    """
    obs_dates = obs['SFC'].keys()
    if verif is not None:
        verif_dates = verif.keys()
    # For BUFR dates, find for all models
    bufr_dates_list = [bufr['SFC'][key].keys() for key in bufr['SFC'].keys()]
    bufr_dates = bufr_dates_list[0]
    for m in range(1, len(bufr_dates_list)):
        bufr_dates = set(bufr_dates).intersection(set(bufr_dates_list[m]))
    if verif is not None:
        all_dates = (set(verif_dates).intersection(set(obs_dates))).intersection(bufr_dates)
    else:
        all_dates = set(obs_dates).intersection(bufr_dates)
    if len(all_dates) == 0:
        raise ValueError('Sorry, no matching dates found in data!')
    print('Found %d matching dates.' % len(all_dates))
    if return_data:
        for lev in ['SFC', 'PROF', 'DAY']:
            for model in bufr[lev].keys():
                for date in bufr[lev][model].keys():
                    if date not in all_dates:
                        bufr[lev][model].pop(date, None)
        for date in obs_dates:
            if date not in all_dates:
                obs['SFC'].pop(date, None)
                obs['SNDG'].pop(date, None)
        if verif is not None:
            for date in verif_dates:
                if date not in all_dates:
                    verif.pop(date, None)
        return bufr, obs, verif, sorted(list(all_dates))
    else:
        return sorted(list(all_dates))


def get_array(dictionary):
    """
    Transforms a nested dictionary into an nd numpy array, assuming that each nested sub-dictionary has the same
    structure and that the values elements of the innermost dictionary is either a list or a float value. Function
    _get_array is its recursive sub-function.

    :param dictionary:
    :return:
    """
    dim_list = []
    d = dictionary
    while isinstance(d, dict):
        dim_list.append(len(d.keys()))
        d = d.values()[0]
    try:
        dim_list.append(len(d))
    except:
        pass
    out_array = np.full(dim_list, np.nan, dtype=np.float64)
    _get_array(dictionary, out_array)
    return out_array


def _get_array(dictionary, out_array):
    if dictionary == {}:  # in case there's an empty dict for any reason
        return
    if isinstance(dictionary.values()[0], list):
        for i, L in enumerate(dictionary.values()):
            out_array[i, :] = np.asarray(L)
    elif isinstance(dictionary.values()[0], float):
        for i, L in enumerate(dictionary.values()):
            out_array[i] = L
    else:
        for i, d in enumerate(dictionary.values()):
            _get_array(d, out_array[i, :])


def unpickle(bufr_file, obs_file, verif_file):
    """
    Shortcut function to unpickle bufr, obs, and verif files all at once. verif_file may be None if running the model.

    :param bufr_file: str: full path to pickled BUFR data file
    :param obs_file: str: full path to pickled OBS data file
    :param verif_file: str: full path to pickled VERIFICATION data file
    :return:
    """
    print('Loading BUFKIT data from %s...' % bufr_file)
    with open(bufr_file, 'rb') as handle:
        bufr = pickle.load(handle)
    print('Loading OBS data from %s...' % obs_file)
    with open(obs_file, 'rb') as handle:
        obs = pickle.load(handle)
    if verif_file is not None:
        print('Loading VERIFICATION data from %s...' % verif_file)
        with open(verif_file, 'rb') as handle:
            verif = pickle.load(handle)
    else:
        verif = None
    return bufr, obs, verif


def get_ghcn_stid(config, stid):
    """
    After code by Luke Madaus.

    Gets the GHCN station ID from the 4-letter station ID.
    """
    main_addr = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa'

    site_directory = config['SITE_ROOT']
    # Check to see that ish-history.txt exists
    stations_file = 'isd-history.txt'
    stations_filename = '%s/%s' % (site_directory, stations_file)
    if not os.path.exists(stations_filename):
        print('get_ghcn_stid: downloading site name database')
        try:
            from urllib2 import urlopen
            response = urlopen('%s/%s' % (main_addr, stations_file))
            with open(stations_filename, 'w') as f:
                f.write(response.read())
        except BaseException as e:
            print('get_ghcn_stid: unable to download site name database')
            print("*** Reason: '%s'" % str(e))

    # Now open this file and look for our siteid
    site_found = False
    infile = open(stations_filename, 'r')
    station_wbans = []
    station_ghcns = []
    for line in infile:
        if stid.upper() in line:
            linesp = line.split()
            if (not linesp[0].startswith('99999') and not site_found
                    and not linesp[1].startswith('99999')):
                try:
                    site_wban = int(linesp[0])
                    station_ghcn = int(linesp[1])
                    # site_found = True
                    print('get_ghcn_stid: site found for %s (%s)' %
                          (stid, station_ghcn))
                    station_wbans.append(site_wban)
                    station_ghcns.append(station_ghcn)
                except:
                    continue
    if len(station_wbans) == 0:
        raise ValueError('get_ghcn_stid error: so station found for %s' % stid)

    # Format station as USW...
    usw_format = 'USW000%05d'
    return usw_format % station_ghcns[0]


# ==================================================================================================================== #
# Conversion functions
# ==================================================================================================================== #

def dewpoint(T, RH):
    """
    Calculates dewpoint from T in Fahrenheit and RH in percent.
    """

    def FtoC(T):
        return (T - 32.) / 9. * 5.

    def CtoF(T):
        return 9. / 5. * T + 32.

    b = 17.67
    c = 243.5  # deg C

    def gamma(T, RH):
        return np.log(RH/100.) + b * T/ (c + T)

    T = FtoC(T)
    TD = c * gamma(T, RH) / (b - gamma(T, RH))
    return CtoF(TD)
