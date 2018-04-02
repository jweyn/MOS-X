#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for processing OBS data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from collections import OrderedDict
from mosx.MesoPy import Meso
from metpy.io import get_upper_air_data
from metpy.calc import interp
from mosx.util import generate_dates, get_array


def upper_air(config, date, use_nan_sounding=False, use_existing=True, save=True):
    """
    Retrieves upper-air data and interpolates to pressure levels. If use_nan_sounding is True, then if a retrieval
    error occurs, a blank sounding will be returned instead of an error.

    :param config:
    :param date: datetime
    :param use_nan_sounding: bool: if True, use sounding of NaNs instead of raising an error
    :param use_existing: bool: preferentially use existing soundings in sounding_data_dir
    :param save: bool: if True, save processed soundings to sounding_data_dir
    :return:
    """
    variables = ['height', 'temperature', 'dewpoint', 'u_wind', 'v_wind']

    # Define levels for interpolation: same as model data, except omitting lowest_p_level
    plevs = [600, 750, 850, 925]
    pres_interp = [p for p in plevs if p <= config['lowest_p_level']]

    # Try retrieving the sounding, first checking for existing
    if config['verbose']:
        print('  Retrieving sounding for %s...' % datetime.strftime(date, '%Y%m%d%H'))
    nan_sounding = False
    retrieve_sounding = False
    sndg_data_dir = config['Obs']['sounding_data_dir']
    if not(os.path.isdir(sndg_data_dir)):
        os.makedirs(sndg_data_dir)
    sndg_file = '%s/%s_SNDG_%s.pkl' % (sndg_data_dir, config['station_id'], datetime.strftime(date, '%Y%m%d%H'))
    if use_existing:
        try:
            with open(sndg_file, 'rb') as handle:
                data = pickle.load(handle)
            if config['verbose']:
                print('    Read from file.')
        except:
            retrieve_sounding = True
    else:
        retrieve_sounding = True
    if retrieve_sounding:
        try:
            dset = get_upper_air_data(date, config['Obs']['sounding_station_id'])
        except:
            # Try again
            try:
                dset = get_upper_air_data(date, config['Obs']['sounding_station_id'])
            except:
                if use_nan_sounding:
                    if config['verbose']:
                        print('    Warning: unable to retrieve sounding; using nan.')
                    nan_sounding = True
                else:
                    raise ValueError('error retrieving sounding for %s' % date)

        # Retrieve pressure for interpolation to fixed levels
        if not nan_sounding:
            pressure = dset.variables['pressure']
            pres = np.array([p.magnitude for p in list(pressure)])  # units are hPa

        # Get variables and interpolate; add to dictionary
        data = OrderedDict()
        for var in variables:
            if not nan_sounding:
                var_data = dset.variables[var]
                var_array = np.array([v.magnitude for v in list(var_data)])
                var_interp = interp(pres_interp, pres, var_array)
                data[var] = var_interp.tolist()
            else:
                data[var] = [np.nan] * len(pres_interp)

        # Save
        if save and not nan_sounding:
            with open(sndg_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def get_obs_hourly(config, api_dates, vars_api, units, return_minute=False):
    """
    Retrieve hourly obs data in a pd dataframe. In order to ensure that there is no missing hourly indices, use
    dataframe.reindex on each retrieved dataframe.

    :param api_dates: dates from generate_dates
    :param vars_api: str: string formatted for api call var parameter
    :param units: str: string formatted for api call units parameter
    :param return_minute: bool: if True, return the minute of hourly obs
    :return: pd.DataFrame: formatted hourly obs DataFrame
    """
    # Initialize Meso
    m = Meso(token=config['meso_token'])
    if config['verbose']:
        print('MesoPy initialized for station %s' % config['station_id'])

    # Retrieve data
    obs_final = pd.DataFrame()
    for api_date in api_dates:
        if config['verbose']:
            print('Retrieving data from %s to %s...' % api_date)
        obs = m.timeseries(stid=config['station_id'], start=api_date[0], end=api_date[1], vars=vars_api, units=units,
                           hfmetars='0')
        obspd = pd.DataFrame.from_dict(obs['STATION'][0]['OBSERVATIONS'])

        # Rename columns to requested vars
        obs_var_names = obs['STATION'][0]['SENSOR_VARIABLES']
        obs_var_keys = list(obs_var_names.keys())
        col_names = list(map(''.join, obspd.columns.values))
        for c in range(len(col_names)):
            col = col_names[c]
            for k in range(len(obs_var_keys)):
                key = obs_var_keys[k]
                if col == list(obs_var_names[key].keys())[0]:
                    col_names[c] = key
        obspd.columns = col_names

        # Change datetime column to datetime object
        dateobj = pd.to_datetime(obspd['date_time'])
        obspd['date_time'] = dateobj
        datename = 'date_time'
        obspd = obspd.rename(columns={'date_time': datename})

        # Reformat data into hourly obs
        # Find mode of minute data: where the hourly metars are
        if config['verbose']:
            print('Finding METAR observation times...')
        minutes = []
        for row in obspd.iterrows():
            date = row[1][datename]
            minutes.append(date.minute)
        minute_count = np.bincount(np.array(minutes))
        rev_count = minute_count[::-1]
        minute_mode = minute_count.size - rev_count.argmax() - 1

        if config['verbose']:
            print('Finding hourly data...')
        obs_hourly = obspd[pd.DatetimeIndex(obspd[datename]).minute == minute_mode]
        obs_hourly = obs_hourly.set_index(datename)

        # May not have precip if none is recorded
        try:
            obs_hourly['precip_accum_one_hour'].fillna(0.0, inplace=True)
        except KeyError:
            obs_hourly['precip_accum_one_hour'] = 0.0

        # Need to reorder the column names
        obs_hourly.sort_index(axis=1, inplace=True)

        # Finally, re-index by hourly. Fills missing with NaNs. Try to interpolate the NaNs.
        expected_start = datetime.strptime(api_date[0], '%Y%m%d%H%M').replace(minute=minute_mode)
        expected_end = datetime.strptime(api_date[1], '%Y%m%d%H%M').replace(minute=minute_mode)
        expected_times = pd.date_range(expected_start, expected_end, freq='H').to_pydatetime()
        obs_hourly = obs_hourly.reindex(expected_times)
        obs_hourly = obs_hourly.interpolate(limit=3)

        obs_final = pd.concat((obs_final, obs_hourly))

    if return_minute:
        return obs_final, minute_mode
    else:
        return obs_final


def obs(config, output_file=None, num_hours=24, interval=3, use_nan_sounding=False, use_existing_sounding=True):
    """
    Generates observation data from MesoWest and UCAR soundings and saves to a file, which can later be retrieved for
    either training data or model run data.

    :param config:
    :param output_file: str: output file path
    :param num_hours: int: number of hours to retrieve obs
    :param interval: int: retrieve obs every 'interval' hours
    :param use_nan_sounding: bool: if True, uses a sounding of NaNs rather than omitting a day if sounding is missing
    :param use_existing_sounding: bool: if True, preferentially uses saved soundings in sounding_data_dir
    :return:
    """
    if output_file is None:
        output_file = '%s/%s_obs.pkl' % (config['SITE_ROOT'], config['station_id'])

    start_date = datetime.strptime(config['data_start_date'], '%Y%m%d') - timedelta(hours=num_hours)
    dates = generate_dates(config)
    api_dates = generate_dates(config, api=True, start_date=start_date)

    # Look for desired variables
    vars_request = ['air_temp', 'altimeter', 'precip_accum_one_hour', 'relative_humidity',
                    'wind_speed', 'wind_direction']

    # Add variables to the api request
    vars_api = ''
    for var in vars_request:
        vars_api += var + ','
    vars_api = vars_api[:-1]

    # Units
    units = 'temp|f,precip|in,speed|kts'

    # Retrieve station data
    obs_hourly = get_obs_hourly(config, api_dates, vars_api, units)

    # Retrieve upper-air sounding data
    if config['verbose']:
        print('Retrieving upper-air sounding data...')
    soundings = OrderedDict()
    if config['Obs']['use_soundings']:
        for date in dates:
            soundings[date] = OrderedDict()
            start_date = date - timedelta(days=1)  # get the previous day's soundings
            for hour in [0, 12]:
                sounding_date = start_date + timedelta(hours=hour)
                try:
                    sounding = upper_air(sounding_date, use_nan_sounding, use_existing=use_existing_sounding)
                    soundings[date][sounding_date] = sounding
                except:
                    print('Warning: problem retrieving soundings for %s' % datetime.strftime(date, '%Y%m%d'))
                    soundings.pop(date)
                    break

    # Create dictionary of days
    if config['verbose']:
        print('Converting to output dictionary...')
    obs_export = OrderedDict({'SFC': OrderedDict(),
                              'SNDG': OrderedDict()})
    for date in dates:
        if config['Obs']['use_soundings'] and date not in soundings.keys():
            continue
        # Need to ensure we use the right intervals to have 22:5? Z obs
        start = pd.Timestamp((date - timedelta(hours=num_hours - interval + 2)))
        end = pd.Timestamp((date - timedelta(hours=1)))
        obs_export['SFC'][date] = OrderedDict(obs_hourly.loc[start:end:interval].to_dict(into=OrderedDict))
        if config['Obs']['use_soundings']:
            obs_export['SNDG'][date] = soundings[date]

    # Export final data
    if config['verbose']:
        print('-> Exporting to %s' % output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(obs_export, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def process(config, obs):
    """
    Returns a numpy array of obs for use in mosx_predictors. The first dimension is date; all other dimensions are
    serialized.

    :param config:
    :param obs: dict: dictionary of processed obs data
    :return:
    """
    if config['verbose']:
        print('Processing array for obs data...')

    # Surface observations
    sfc = obs['SFC']
    num_days = len(sfc.keys())
    variables = sorted(sfc[sfc.keys()[0]].keys())
    num_vars = len(variables)
    num_times = len(sfc[sfc.keys()[0]][variables[0]])
    sfc_array = get_array(sfc)
    sfc_array_r = np.reshape(sfc_array, (num_days, num_vars * num_times))

    # Sounding observations
    if config['Obs']['use_soundings']:
        sndg_array = get_array(obs['SNDG'])
        # num_days should be the same first dimension
        sndg_shape = (num_days, sndg_array.size / num_days)
        sndg_array_r = np.reshape(sndg_array, sndg_shape)
        return np.hstack((sfc_array_r, sndg_array_r))
    else:
        return sfc_array_r
