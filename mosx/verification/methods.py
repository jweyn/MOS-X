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
import requests
from collections import OrderedDict
from mosx.MesoPy import Meso
from mosx.obs.methods import get_obs_hourly
from mosx.util import generate_dates, get_array


def get_cf6_files(config, num_files=1):
    """
    After code by Luke Madaus

    Retrieves CF6 climate verification data released by the NWS. Parameter num_files determines how many recent files
    are downloaded.
    """

    # Create directory if it does not exist
    site_directory = config['SITE_ROOT']

    # Construct the web url address. Check if a special 3-letter station ID is provided.
    nws_url = 'http://forecast.weather.gov/product.php?site=NWS&issuedby=%s&product=CF6&format=TXT'
    try:
        stid3 = config['station_id3']
    except KeyError:
        stid3 = config['station_id'][1:].upper()
    nws_url = nws_url % stid3

    # Determine how many files (iterations of product) we want to fetch
    if num_files == 1:
        if config['verbose']:
            print('get_cf6_files: retrieving latest CF6 file for %s' % config['station_id'])
    else:
        if config['verbose']:
            print('get_cf6_files: retrieving %s archived CF6 files for %s' % (num_files, config['station_id']))

    # Fetch files
    for r in range(1, num_files + 1):
        # Format the web address: goes through 'versions' on NWS site which correspond to increasingly older files
        version = 'version=%d&glossary=0' % r
        nws_site = '&'.join((nws_url, version))
        response = requests.get(nws_site)
        cf6_data = response.text

        # Remove the header
        try:
            body_and_footer = cf6_data.split('CXUS')[1]  # Mainland US
        except IndexError:
            try:
                body_and_footer = cf6_data.split('CXHW')[1]  # Hawaii
            except IndexError:
                body_and_footer = cf6_data.split('CXAK')[1]  # Alaska
        body_and_footer_lines = body_and_footer.splitlines()
        if len(body_and_footer_lines) <= 2:
            body_and_footer = cf6_data.split('000')[2]

        # Remove the footer
        body = body_and_footer.split('[REMARKS]')[0]

        # Find the month and year of the file
        current_year = re.search('YEAR: *(\d{4})', body).groups()[0]
        try:
            current_month = re.search('MONTH: *(\D{3,9})', body).groups()[0]
            current_month = current_month.strip()  # Gets rid of newlines and whitespace
            datestr = '%s %s' % (current_month, current_year)
            file_date = datetime.strptime(datestr, '%B %Y')
        except:  # Some files have a different formatting, although this may be fixed now.
            current_month = re.search('MONTH: *(\d{2})', body).groups()[0]
            current_month = current_month.strip()
            datestr = '%s %s' % (current_month, current_year)
            file_date = datetime.strptime(datestr, '%m %Y')

        # Write to a temporary file, check if output file exists, and if so, make sure the new one has more data
        datestr = file_date.strftime('%Y%m')
        filename = '%s/%s_%s.cli' % (site_directory, config['station_id'].upper(), datestr)
        temp_file = '%s/temp.cli' % site_directory
        with open(temp_file, 'w') as out:
            out.write(body)

        def file_len(file_name):
            with open(file_name) as f:
                for i, l in enumerate(f):
                    pass
                return i + 1

        if os.path.isfile(filename):
            old_file_len = file_len(filename)
            new_file_len = file_len(temp_file)
            if old_file_len < new_file_len:
                if config['verbose']:
                    print('get_cf6_files: overwriting %s' % filename)
                os.remove(filename)
                os.rename(temp_file, filename)
            else:
                if config['verbose']:
                    print('get_cf6_files: %s already exists' % filename)
        else:
            if config['verbose']:
                print('get_cf6_files: writing %s' % filename)
            os.rename(temp_file, filename)


def _cf6_wind(config):
    """
    After code by Luke Madaus

    This function is used internally only.

    Generates wind verification values from climate CF6 files stored in SITE_ROOT. These files can be generated
    externally by get_cf6_files.py. This function is not necessary if climo data from _climo_wind is found, except for
    recent values which may not be in the NCDC database yet.

    :param config:
    :return: dict: wind values from CF6 files
    """

    if config['verbose']:
        print('_cf6_wind: searching for CF6 files in %s' % config['SITE_ROOT'])
    allfiles = os.listdir(config['SITE_ROOT'])
    filelist = [f for f in allfiles if f.startswith(config['station_id'].upper()) and f.endswith('.cli')]
    filelist.sort()
    if len(filelist) == 0:
        raise IOError('No CF6 files found.')
    if config['verbose']:
        print('_cf6_wind: found %d CF6 files.' % len(filelist))

    # Interpret CF6 files
    if config['verbose']:
        print('_cf6_wind: reading CF6 files')
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


def _climo_wind(config, dates=None):
    """
     Fetches climatological wind data using ulmo package to retrieve NCDC archives.

    :param config:
    :param dates: list of datetime objects
    :return: dict: dictionary of wind values
    """
    import ulmo

    if config['verbose']:
        print('_climo_wind: fetching data from NCDC (may take a while)...')
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


def pop_rain(series):
    """
    Converts a series of rain values into 0 or 1 depending on whether there is measurable rain
    :param series:
    :return:
    """
    new_series = series.copy()
    new_series[series >= 0.01] = 1.
    new_series[series < 0.01] = 0.
    return new_series


def categorical_rain(series):
    """
    Converts a series of rain values into categorical precipitation quantities a la MOS.
    :param series:
    :return:
    """
    new_series = series.copy()
    for j in range(len(series)):
        if series.iloc[j] < 0.01:
            new_series.iloc[j] = 0.
        elif series.iloc[j] < 0.10:
            new_series.iloc[j] = 1.
        elif series.iloc[j] < 0.25:
            new_series.iloc[j] = 2.
        elif series.iloc[j] < 0.50:
            new_series.iloc[j] = 3.
        elif series.iloc[j] < 1.00:
            new_series.iloc[j] = 4.
        elif series.iloc[j] < 2.00:
            new_series.iloc[j] = 5.
        elif series.iloc[j] >= 2.00:
            new_series.iloc[j] = 6.
        else:  # missing, or something else that's strange
            new_series.iloc[j] = 0.
    return new_series


def verification(config, output_file=None, use_cf6=True, use_climo=True, force_rain_quantity=False):
    """
    Generates verification data from MesoWest and saves to a file, which is used to train the model and check test
    results.

    :param config:
    :param output_file: str: path to output file
    :param use_cf6: bool: if True, uses wind values from CF6 files
    :param use_climo: bool: if True, uses wind values from NCDC climatology
    :param force_rain_quantity: if True, returns the actual quantity of rain (rather than POP); useful for validation
    files
    :return:
    """
    if output_file is None:
        output_file = '%s/%s_verif.pkl' % (config['SITE_ROOT'], config['station_id'])

    dates = generate_dates(config)
    api_dates = generate_dates(config, api=True, api_add_hour=config['forecast_hour_start'] + 24)

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
        obs_hourly_verify = get_obs_hourly(config, api_dates, vars_api, units)

    # Read new data for daily values
    m = Meso(token=config['meso_token'])

    if config['verbose']:
        print('verification: MesoPy initialized for station %s' % config['station_id'])
        print('verification: retrieving latest obs and metadata')
    latest = m.latest(stid=config['station_id'])
    obs_list = list(latest['STATION'][0]['SENSOR_VARIABLES'].keys())

    # Look for desired variables
    vars_request = ['air_temp', 'wind_speed', 'precip_accum_one_hour']
    vars_option = ['air_temp_low_6_hour', 'air_temp_high_6_hour', 'precip_accum_six_hour']

    # Add variables to the api request if they exist
    if config['verbose']:
        print('verification: searching for 6-hourly variables...')
    for var in vars_option:
        if var in obs_list:
            if config['verbose']:
                print('verification: found variable %s, adding to data' % var)
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
            print('verification: retrieving data from %s to %s' % api_date)
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

    # Make sure we have columns for all requested variables
    for var in vars_request:
        if var not in col_names:
            obspd = obspd.assign(**{var: np.nan})

    # Change datetime column to datetime object, subtract 6 hours to use 6Z days
    if config['verbose']:
        print('verification: setting time back %d hours for daily statistics' % config['forecast_hour_start'])
    dateobj = pd.to_datetime(obspd['date_time']) - timedelta(hours=config['forecast_hour_start'])
    obspd['date_time'] = dateobj
    datename = 'date_time_minus_%d' % config['forecast_hour_start']
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
        print('verification: grouping data by hour for hourly observations')
    # Note that obs in hour H are reported at hour H, not H+1
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
        print('verification: grouping data by day for daily verifications')
    obs_daily = obs_hourly.groupby([pd.DatetimeIndex(obs_hourly[datename]).year,
                                    pd.DatetimeIndex(obs_hourly[datename]).month,
                                    pd.DatetimeIndex(obs_hourly[datename]).day]).agg(aggregate)

    if config['verbose']:
        print('verification: checking matching dates for daily obs and CF6')
    if use_climo:
        try:
            climo_values = _climo_wind(config, dates)
        except BaseException as e:
            if config['verbose']:
                print("verification: warning: '%s' while reading climo data" % str(e))
            climo_values = {}
    else:
        if config['verbose']:
            print('verification: not using climo.')
        climo_values = {}
    if use_cf6:
        try:
            get_cf6_files(config)
        except BaseException as e:
            if config['verbose']:
                print("verification: warning: '%s' while getting CF6 files" % str(e))
        try:
            cf6_values = _cf6_wind(config)
        except BaseException as e:
            if config['verbose']:
                print("verification: warning: '%s' while reading CF6 files" % str(e))
            cf6_values = {}
    else:
        if config['verbose']:
            print('verification: not using CF6.')
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
                    print('verification: warning: obs wind for %s much larger than wind from cf6/climo; using obs' % 
                          date)
                else:
                    obs_daily.loc[index, 'wind_speed'] = cf6_wind
            else:
                count_rows -= 1
    if config['verbose']:
        print('verification: found %d matching rows.' % count_rows)

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
        print('verification: generating final verification dictionary...')
    if 'air_temp_high_6_hour' in vars_request:
        obs_daily.rename(columns={'air_temp_high_6_hour': 'Tmax'}, inplace=True)
    else:
        obs_daily.rename(columns={'air_temp_max': 'Tmax'}, inplace=True)
    if 'air_temp_low_6_hour' in vars_request:
        obs_daily.rename(columns={'air_temp_low_6_hour': 'Tmin'}, inplace=True)
    else:
        obs_daily.rename(columns={'air_temp_min': 'Tmin'}, inplace=True)
    if 'precip_accum_six_hour' in vars_request:
        obs_daily.rename(columns={'precip_accum_six_hour': 'Rain'}, inplace=True)
    else:
        obs_daily.rename(columns={'precip_accum_one_hour': 'Rain'}, inplace=True)
    obs_daily.rename(columns={'wind_speed': 'Wind'}, inplace=True)

    # Deal with the rain depending on the type of forecast requested
    obs_daily['Rain'].fillna(0.0, inplace=True)
    if config['Model']['rain_forecast_type'] == 'pop' and not force_rain_quantity:
        obs_daily.loc[:, 'Rain'] = pop_rain(obs_daily['Rain'])
    elif config['Model']['rain_forecast_type'] == 'categorical' and not force_rain_quantity:
        obs_daily.loc[:, 'Rain'] = categorical_rain(obs_daily['Rain'])

    # Set the date time index
    obs_daily = obs_daily.rename(columns={datename: 'date_time'})
    obs_daily = obs_daily.set_index('date_time')

    # Export final data
    if config['verbose']:
        print('verification: -> exporting to %s' % output_file)
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
            if config['verbose']:
                print('verification: warning: omitting day %s; missing data' % date)
            continue  # No verification can have missing values
        if config['Model']['predict_timeseries']:
            start = pd.Timestamp((date + timedelta(hours=5)))
            end = pd.Timestamp((date + timedelta(hours=30)))
            try:
                series = obs_hourly_verify.loc[start:end]
            except KeyError:
                # No values for the day
                if config['verbose']:
                    print('verification: warning: omitting day %s; missing data' % date)
                continue
            if series.isnull().values.any():
                if config['verbose']:
                    print('verification: warning: omitting day %s; missing data' % date)
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
            print('verification.process: processing array for verification data')
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
