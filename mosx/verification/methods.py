#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for processing VERIFICATION data.
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from collections import OrderedDict
from mosx.MesoPy import Meso
from mosx.obs.methods import get_obs_hourly
from mosx.util import generate_dates, get_array


def cf6_parser(config):
    """
    After code by Luke Madaus

    This function is used internally only.

    Generates wind verification values from climate CF6 files stored in SITE_ROOT. These files can be generated
    externally by get_cf6_files.py. This function is not necessary if climo data from climo_wind is found, except for
    recent values which may not be in the NCDC database yet.

    :param config:
    :return: dict: wind values from CF6 files
    """

    if config['verbose']:
        print('Searching for CF6 files in %s...' % config['SITE_ROOT'])
    allfiles = os.listdir(config['SITE_ROOT'])
    filelist = [f for f in allfiles if f.startswith(config['station_id'].upper()) and f.endswith('.cli')]
    filelist.sort()
    if len(filelist) == 0:
        raise IOError('No CF6 files found.')
    if config['verbose']:
        print('  Found %d CF6 files.' % len(filelist))

    # Interpret CF6 files
    if config['verbose']:
        print('Reading CF6 files...')
    cf6_values = {}
    for file in filelist:
        year, month = re.search('(\d{4})(\d{2})', file).groups()
        infile = open('%s/%s' % (config['SITE_ROOT'], file), 'r')
        for line in infile:
            matcher = re.compile(
                '( \d|\d{2}) ( \d{2}|-\d{2}|  \d| -\d|\d{3})')
            if matcher.match(line):
                # We've found an ob line!
                lsp = line.split()
                day = int(lsp[0])
                curdt = datetime(int(year), int(month), day)
                cf6_values[curdt] = {}
                # Wind
                if lsp[11] == 'M':
                    cf6_values[curdt]['wind'] = 0.0
                else:
                    cf6_values[curdt]['wind'] = float(lsp[11]) * 0.868976

    return cf6_values


def climo_wind(config, dates=None):
    """
     Fetches climatological wind data using ulmo package to retrieve NCDC archives.

    :param config:
    :param dates: list of datetime objects
    :return: dict: dictionary of wind values
    """
    import ulmo

    if config['verbose']:
        print('Climo: fetching data from NCDC (may take a while)...')
    v = 'WSF2'
    wind_dict = {}
    try:
        D = ulmo.ncdc.ghcn_daily.get_data(config['climo_station_id'], as_dataframe=True, elements=[v])
    except:
        return wind_dict

    if dates is None:
        dates = list(D[v].index.to_timestamp().to_pydatetime())
    for date in dates:
        wind_dict[date] = {'wind': D[v].loc[date]['value'] / 10. * 1.94384}

    return wind_dict


def verification(config, output_file=None, use_cf6=True, use_climo=True,):
    """
    Generates verification data from MesoWest and saves to a file, which is used to train the model and check test
    results.

    :param config:
    :param output_file: str: path to output file
    :param use_cf6: bool: if True, uses wind values from CF6 files
    :param use_climo: bool: if True, uses wind values from NCDC climatology
    :return:
    """
    if output_file is None:
        output_file = '%s/%s_verif.pkl' % (config['SITE_ROOT'], config['station_id'])

    end_date = datetime.strptime(config['data_end_date'], '%Y%m%d') + timedelta(days=1)
    dates = generate_dates(config)
    api_dates = generate_dates(config, api=True, end_date=end_date, api_end_hour=6)

    # If a time series is desired, then get hourly data
    if config['Model']['predict_timeseries']:

        # Look for desired variables
        vars_request = ['air_temp', 'relative_humidity', 'wind_speed', 'precip_accum_one_hour']

        # Add variables to the api request
        vars_api = ''
        for var in vars_request:
            vars_api += var + ','
        vars_api = vars_api[:-1]

        # Units
        units = 'temp|f,precip|in,speed|kts'

        # Retrieve data
        obs_hourly_verify = get_obs_hourly(api_dates, vars_api, units)

    # Read new data for daily values
    m = Meso(token=config['meso_token'])

    if config['verbose']:
        print('MesoPy initialized for station %s' % config['station_id'])
        print('Retrieving latest obs and metadata...')
    latest = m.latest(stid=config['station_id'])
    obs_list = list(latest['STATION'][0]['SENSOR_VARIABLES'].keys())

    # Look for desired variables
    vars_request = ['air_temp', 'wind_speed', 'precip_accum_one_hour']
    vars_option = ['air_temp_low_6_hour', 'air_temp_high_6_hour', 'precip_accum_six_hour']

    # Add variables to the api request if they exist
    if config['verbose']:
        print('Searching for 6-hourly variables...')
    for var in vars_option:
        if var in obs_list:
            if config['verbose']:
                print('  Found variable %s, adding to data...' % var)
            vars_request += [var]
    vars_api = ''
    for var in vars_request:
        vars_api += var + ','
    vars_api = vars_api[:-1]

    # Units
    units = 'temp|f,precip|in,speed|kts'

    # Retrieve data
    obspd = pd.DataFrame()
    for api_date in api_dates:
        if config['verbose']:
            print('Retrieving data from %s to %s...' % api_date)
        obs = m.timeseries(stid=config['station_id'], start=api_date[0], end=api_date[1], vars=vars_api, units=units)
        obspd = pd.concat((obspd, pd.DataFrame.from_dict(obs['STATION'][0]['OBSERVATIONS'])), ignore_index=True)

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

    # Change datetime column to datetime object, subtract 6 hours to use 6Z days
    if config['verbose']:
        print('Setting time back 6 hours to analyze daily 6Z--6Z statistics...')
    dateobj = pd.to_datetime(obspd['date_time']) - timedelta(hours=6)
    obspd['date_time'] = dateobj
    datename = 'date_time_minus_6'
    obspd = obspd.rename(columns={'date_time': datename})

    # Reformat data into hourly and daily
    # Hourly
    def hour(dates):
        date = dates.iloc[0]
        return datetime(date.year, date.month, date.day, date.hour)

    def last(values):
        return values.iloc[-1]

    aggregate = {datename: hour}
    if 'air_temp_high_6_hour' in vars_request and 'air_temp_low_6_hour' in vars_request:
        aggregate['air_temp_high_6_hour'] = np.max
        aggregate['air_temp_low_6_hour'] = np.min
    aggregate['air_temp'] = {'air_temp_max': np.max, 'air_temp_min': np.min}
    if 'precip_accum_six_hour' in vars_request:
        aggregate['precip_accum_six_hour'] = np.max
    aggregate['wind_speed'] = np.max
    aggregate['precip_accum_one_hour'] = np.max

    if config['verbose']:
        print('Grouping data by hour for hourly observations...')
        print('  (Note that obs in hour H are reported at hour H, not H+1)')
    obs_hourly = obspd.groupby([pd.DatetimeIndex(obspd[datename]).year,
                                pd.DatetimeIndex(obspd[datename]).month,
                                pd.DatetimeIndex(obspd[datename]).day,
                                pd.DatetimeIndex(obspd[datename]).hour]).agg(aggregate)
    # Rename columns
    col_names = obs_hourly.columns.values
    col_names_new = []
    for c in range(len(col_names)):
        if col_names[c][0] == 'air_temp':
            col_names_new.append(col_names[c][1])
        else:
            col_names_new.append(col_names[c][0])

    obs_hourly.columns = col_names_new

    # Daily
    def day(dates):
        date = dates.iloc[0]
        return datetime(date.year, date.month, date.day)

    aggregate[datename] = day
    aggregate['air_temp_min'] = np.min
    aggregate['air_temp_max'] = np.max
    aggregate['precip_accum_six_hour'] = np.sum
    try:
        aggregate.pop('air_temp')
    except:
        pass

    if config['verbose']:
        print('Grouping data by day for daily verifications...')
    obs_daily = obs_hourly.groupby([pd.DatetimeIndex(obs_hourly[datename]).year,
                                    pd.DatetimeIndex(obs_hourly[datename]).month,
                                    pd.DatetimeIndex(obs_hourly[datename]).day]).agg(aggregate)

    if config['verbose']:
        print('Checking matching dates for daily obs and CF6...')
    if use_climo:
        try:
            climo_values = climo_wind(dates)
        except:
            if config['verbose']:
                print('  Warning: problem reading climo data.')
            climo_values = {}
    else:
        if config['verbose']:
            print('  Not using climo.')
        climo_values = {}
    if use_cf6:
        try:
            cf6_values = cf6_parser()
        except:
            if config['verbose']:
                print('  Warning: no CF6 files found.')
            cf6_values = {}
    else:
        if config['verbose']:
            print('  Not using CF6.')
        cf6_values = {}
    climo_values.update(cf6_values)  # CF6 has precedence
    count_rows = 0
    for index, row in obs_daily.iterrows():
        date = row[datename]
        if date in climo_values.keys():
            count_rows += 1
            obs_wind = row['wind_speed']
            cf6_wind = climo_values[date]['wind']
            if not (np.isnan(cf6_wind)):
                if obs_wind - cf6_wind >= 5:
                    print('  Warning: obs wind for %s much larger than wind from cf6/climo; using obs' % date)
                else:
                    obs_daily.loc[index, 'wind_speed'] = cf6_wind
            else:
                count_rows -= 1
    if config['verbose']:
        print('  Found %d matching rows.' % count_rows)

    # Round
    round_dict = {'wind_speed': 0}
    if 'air_temp_high_6_hour' in vars_request:
        round_dict['air_temp_high_6_hour'] = 0
    if 'air_temp_low_6_hour' in vars_request:
        round_dict['air_temp_low_6_hour'] = 0
    round_dict['air_temp_max'] = 0
    round_dict['air_temp_min'] = 0
    if 'precip_accum_six_hour' in vars_request:
        round_dict['precip_accum_six_hour'] = 2
    round_dict['precip_accum_one_hour'] = 2
    obs_daily = obs_daily.round(round_dict)

    # Generation of final output data
    if config['verbose']:
        print('Generating final verification dictionary...')
    if 'air_temp_high_6_hour' in vars_request:
        obs_daily.rename(columns={'air_temp_high_6_hour': 'Tmax'}, inplace=True)
        print('  Set column air_temp_high_6_hour to Tmax')
    else:
        obs_daily.rename(columns={'air_temp_max': 'Tmax'}, inplace=True)
        print('  Set column air_temp_max to Tmax')
    if 'air_temp_low_6_hour' in vars_request:
        obs_daily.rename(columns={'air_temp_low_6_hour': 'Tmin'}, inplace=True)
        print('  Set column air_temp_low_6_hour to Tmin')
    else:
        obs_daily.rename(columns={'air_temp_min': 'Tmin'}, inplace=True)
        print('  Set column air_temp_min to Tmin')
    if 'precip_accum_six_hour' in vars_request:
        obs_daily.rename(columns={'precip_accum_six_hour': 'Rain'}, inplace=True)
        print('  Set column precip_accum_six_hour to Rain')
    else:
        obs_daily.rename(columns={'precip_accum_one_hour': 'Rain'}, inplace=True)
        print('  Set column precip_accum_one_hour to Rain')
    obs_daily.rename(columns={'wind_speed': 'Wind'}, inplace=True)
    print('  Set column wind_speed to Wind')

    obs_daily['Rain'].fillna(0.0, inplace=True)
    obs_daily = obs_daily.rename(columns={datename: 'date_time'})
    obs_daily = obs_daily.set_index('date_time')

    # Export final data
    if config['verbose']:
        print('-> Exporting to %s' % output_file)
    export_cols = ['Tmax', 'Tmin', 'Wind', 'Rain']
    for col in obs_daily.columns:
        if col not in export_cols:
            obs_daily.drop(col, 1, inplace=True)

    export_dict = OrderedDict()
    for date in dates:
        try:
            day_dict = obs_daily.loc[date].to_dict(into=OrderedDict)
        except KeyError:
            continue
        if np.any(np.isnan(day_dict.values())):
            continue  # No verification can have missing values
        if config['Model']['predict_timeseries']:
            start = pd.Timestamp((date + timedelta(hours=5)))  # ensures we have obs a few minutes before 6Z
            end = pd.Timestamp((date + timedelta(hours=30)))
            series = obs_hourly_verify.loc[start:end]
            if series.isnull().values.any():
                continue
            series_dict = OrderedDict(series.to_dict(into=OrderedDict))
            day_dict.update(series_dict)
        export_dict[date] = day_dict
    with open(output_file, 'wb') as handle:
        pickle.dump(export_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def process(config, verif):
    """
    Returns a numpy array of verification data for use in mosx_predictors. The first dimension is date, the second is
    variable.

    :param config:
    :param verif: dict: dictionary of processed verification data; may be None
    :return: ndarray: array of processed verification targets
    """
    if verif is not None:
        if config['verbose']:
            print('Processing array for verification data...')
        num_days = len(verif.keys())
        variables = ['Tmax', 'Tmin', 'Wind', 'Rain']
        day_verif_array = np.full((num_days, len(variables)), np.nan, dtype=np.float64)
        for d in range(len(verif.keys())):
            date = verif.keys()[d]
            for v in range(len(variables)):
                var = variables[v]
                day_verif_array[d, v] = verif[date][var]
        if config['Model']['predict_timeseries']:
            hour_verif = OrderedDict(verif)
            for date in hour_verif.keys():
                for variable in variables:
                    hour_verif[date].pop(variable, None)
            variables = sorted(hour_verif[hour_verif.keys()[0]].keys())
            num_vars = len(variables)
            hour_verif_array = get_array(hour_verif)
            hour_verif_array = np.reshape(hour_verif_array, (num_days, num_vars * 25))
            verif_array = np.concatenate((day_verif_array, hour_verif_array), axis=1)
            return verif_array
        else:
            return day_verif_array
    else:
        return None
