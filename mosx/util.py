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
from datetime import datetime
import pickle


# ==================================================================================================================== #
# Classes
# ==================================================================================================================== #

class TimeSeriesEstimator(object):
    """
    Wrapper class for containing separately-trained daily and timeseries estimators.
    """
    def __init__(self, daily_estimator, timeseries_estimator):
        self.array_form = True
        self.daily_estimator = daily_estimator
        self.timeseries_estimator = timeseries_estimator
        self.named_steps = self.daily_estimator.named_steps

    def fit(self, predictor_array, verification_array, **kwargs):
        """
        Fit both the daily and the timeseries estimators.
        :param predictor_array: num_samples x num_features
        :param verification_array: num_samples x num_daily+num_ts
        :param kwargs: kwargs passed to fit methods
        :return:
        """
        print('Fitting DAILY estimator...')
        self.daily_estimator.fit(predictor_array, verification_array[:, :4], **kwargs)
        print('Fitting TIMESERIES estimator...')
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

    # Convert kwargs and ada_boost (kwargs), if available, to int or float types
    config['Model']['kwargs'].walk(walk_kwargs)
    try:
        config['Model']['ada_boost'].walk(walk_kwargs)
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


def generate_dates(config, api=False, start_date=None, end_date=None, api_end_hour=0):
    """
    Returns all of the dates requested from the config. If api is True, then returns a list of (start_date, end_date)
    tuples split by year in strings formatted for the MesoWest API call. If api is False, then returns a list of all
    dates as datetime objects. start_date and end_date are available as options as certain calls require addition of
    some data for prior days.

    :param config:
    :param api: bool: if True, returns dates formatted for MesoWest API call
    :param start_date: str: starting date in config file format (YYYYMMDD)
    :param end_date: str: ending date in config file format (YYYYMMDD)
    :param api_end_hour: int: last hour of last day in an API call, useful for getting up to 6Z data
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
    if start_dt > datetime(start_dt.year, end_dt.month, end_dt.day):
        # Season crosses new year
        end_year = end_dt.year
    else:
        end_year = end_dt.year + 1
    all_dates = []
    if config['is_season']:
        for year in range(start_dt.year, end_year):
            if start_dt > datetime(start_dt.year, end_dt.month, end_dt.day):
                # Season crosses new year
                year2 = year + 1
            else:
                year2 = year
            if api:
                year_start = datetime.strftime(datetime(year, start_dt.month, start_dt.day), '%Y%m%d0000')
                year_end = datetime.strftime(datetime(year2, end_dt.month, end_dt.day, api_end_hour), '%Y%m%d%H00')
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
                    year_end = datetime.strftime(datetime(year, end_dt.month, end_dt.day, api_end_hour), '%Y%m%d%H00')
                else:
                    year_end = datetime.strftime(datetime(year+1, 1, 1, api_end_hour), '%Y%m%d%H00')
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
