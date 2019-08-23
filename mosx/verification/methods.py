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
from mosx.obs.methods import get_obs_hourly, reindex_hourly
from mosx.util import generate_dates, get_array, get_ghcn_stid


def get_cf6_files(config, station_id, num_files=1):
    """
    After code by Luke Madaus
    Retrieves CF6 climate verification data released by the NWS. Parameter num_files determines how many recent files
    are downloaded.
    :param station_id: station ID to obtain cf6 files for
    """

    # Create directory if it does not exist
    site_directory = config['SITE_ROOT']

    # Construct the web url address. Check if a special 3-letter station ID is provided.
    nws_url = 'http://forecast.weather.gov/product.php?site=NWS&issuedby=%s&product=CF6&format=TXT'
    stid3 = station_id[1:].upper()
    nws_url = nws_url % stid3

    # Determine how many files (iterations of product) we want to fetch
    if num_files == 1:
        if config['verbose']:
            print('get_cf6_files: retrieving latest CF6 file for %s' % station_id)
    else:
        if config['verbose']:
            print('get_cf6_files: retrieving %s archived CF6 files for %s' % (num_files, station_id))

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
        filename = '%s/%s_%s.cli' % (site_directory, station_id.upper(), datestr)
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


def _cf6(config, station_id):
    """
    After code by Luke Madaus
    This function is used internally only.
    Generates verification values from climate CF6 files stored in SITE_ROOT. These files can be generated
    externally by get_cf6_files.py. This function is not necessary if climo data from _climo is found, except for
    recent values which may not be in the NCDC database yet.
    :param config:
    :param station_id: station ID to obtain cf6 files for
    :return: dict: wind values from CF6 files
    """

    if config['verbose']:
        print('_cf6: searching for CF6 files in %s' % config['SITE_ROOT'])
    allfiles = os.listdir(config['SITE_ROOT'])
    filelist = [f for f in allfiles if f.startswith(station_id.upper()) and f.endswith('.cli')]
    filelist.sort()
    if len(filelist) == 0:
        raise IOError('No CF6 files found.')
    if config['verbose']:
        print('_cf6: found %d CF6 files.' % len(filelist))

    # Interpret CF6 files
    if config['verbose']:
        print('_cf6: reading CF6 files')
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
                
                # Max temp
                if lsp[1] == 'M':
                    cf6_values[curdt]['max_temp'] = -999.0
                else:
                    cf6_values[curdt]['max_temp'] = float(lsp[1])
                    
                # Min temp
                if lsp[2] == 'M':
                    cf6_values[curdt]['min_temp'] = 999.0
                else:
                    cf6_values[curdt]['min_temp'] = float(lsp[2])
                    
                # Precipitation
                if lsp[7] == 'M':
                    cf6_values[curdt]['precip'] = -999.0
                elif lsp[7] == 'T':
                    cf6_values[curdt]['precip'] = 0
                else:
                    cf6_values[curdt]['precip'] = float(lsp[7])
                    
                # Wind
                if lsp[11] == 'M':
                    cf6_values[curdt]['wind'] = 0.0
                else:
                    cf6_values[curdt]['wind'] = float(lsp[11]) * 0.868976

    return cf6_values


def _climo(config, station_id, dates=None):
    """
     Fetches climatological wind data using ulmo package to retrieve NCDC archives.
    :param config:
    :param station_id: station ID to obtain cf6 files for
    :param dates: list of datetime objects
    :return: dict of high temp, low temp, max wind, and precipitation values
    """
    import ulmo

    if config['verbose']:
        print('_climo: fetching data from NCDC (may take a while)...')
    climo_dict = {}
    D = ulmo.ncdc.ghcn_daily.get_data(get_ghcn_stid(config, station_id), as_dataframe=True, elements=['TMAX','TMIN','WSF2','PRCP'])

    if dates is None:
        dates = list(D['WSF2'].index.to_timestamp().to_pydatetime())
    for date in dates:
        climo_dict[date] = {}
        try:
            climo_dict[date]['max_temp'] = D['TMAX'].loc[date]['value']*0.18+32.0
            climo_dict[date]['min_temp'] = D['TMIN'].loc[date]['value']*0.18+32.0
            climo_dict[date]['wind'] = D['WSF2'].loc[date]['value'] / 10. * 1.94384
            climo_dict[date]['precip'] = D['PRCP'].loc[date]['value'] / 254.0
        except KeyError: #missing data
            if config['verbose']:
                print('_climo: climo data missing for %s',date)
    return climo_dict


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


def verification(config, output_files=None, csv_files=None, use_cf6=True, use_climo=True, force_rain_quantity=False):
    """
    Generates verification data from MesoWest and saves to a file, which is used to train the model and check test
    results.
    :param config:
    :param output_files: str: output file path if just one station, or list of output file paths if multiple stations
    :param csv_files: str: path to csv file containing observations if just one station, or list of paths to csv files if multiple stations
    :param use_cf6: bool: if True, uses wind values from CF6 files
    :param use_climo: bool: if True, uses wind values from NCDC climatology
    :param force_rain_quantity: if True, returns the actual quantity of rain (rather than POP); useful for validation
    files
    :return:
    """
    if config['multi_stations']: #Train on multiple stations
        station_ids = config['station_id']
        if len(station_ids) != len(output_files): #There has to be the same number of output files as station IDs, so raise error if not
            raise ValueError("There must be the same number of output files as station IDs")
        if len(station_ids) != len(csv_files): #There has to be the same number of output files as station IDs, so raise error if not
            raise ValueError("There must be the same number of csv files as station IDs")
    else:
        station_ids = [config['station_id']]
        if output_files is not None:
            output_files = [output_files]
        if csv_files is not None:
            csv_files = [csv_files]
    
    for i in range(len(station_ids)):
        station_id = station_ids[i]
        if output_files is None:
            output_file = '%s/%s_verif.pkl' % (config['SITE_ROOT'], station_id)
        else:
            output_file = output_files[i]
    
        if csv_files is None:
            csv_file = '%s/%s_verif.csv' % (config['SITE_ROOT'], station_id)
        else:
            csv_file = csv_files[i]
    
        dates = generate_dates(config)
        api_dates = generate_dates(config, api=True, api_add_hour=config['forecast_hour_start'] + 24)
        datename = 'date_time_minus_%d' % config['forecast_hour_start']
    
        if config['verbose']:
            print('verification: obtaining observations from csv file')
        with open('%s/%s_obs_vars_request.txt' % (config['SITE_ROOT'], station_id),'rb') as fp:
            vars_request = pickle.load(fp)

        all_obspd = pd.read_csv(csv_file)
        obspd = all_obspd[['date_time']+[vars_request[0]]+[vars_request[2]]+[vars_request[4]]+vars_request[6:]] #subset of data used as verification
        obspd['date_time']=np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in obspd['date_time'].values],dtype='datetime64[s]')
        if config['verbose']:
            print('verification: setting time back %d hours for daily statistics' % config['forecast_hour_start'])
        dateobj = pd.to_datetime(obspd['date_time']) - timedelta(hours=config['forecast_hour_start'])
        obspd['date_time'] = dateobj
        obspd = obspd.rename(columns={'date_time': datename})
        
        # Reformat data into hourly and daily
        # Hourly
        def hour(dates):
            date = dates.iloc[0]
            if type(date) == str: #if data is from csv file, date will be a string instead of a datetime object
                #depending on which version of NumPy or pandas you use, the first or second statement will work
                try:
                    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except:
                    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S+00:00')
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
            if type(date) == str: #if data is from csv file, date will be a string instead of a datetime object
                #depending on which version of NumPy or pandas you use, the first or second statement will work
                try:
                    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except:
                    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S+00:00')
            return datetime(date.year, date.month, date.day)
        
        def min_or_nan(a):
            '''
            Returns the minimum of a 1D array if there are at least 4 non-NaN values, and returns NaN otherwise. This is to ensure 
            having NaNs on days with incomplete data when grouping into daily data rather than incorrect data.
            '''
            if np.count_nonzero(~np.isnan(a)) < 4: #incomplete data
                return np.nan
            else:
                return np.min(a)
                
        def max_or_nan(a):
            '''
            Returns the maximum of a 1D array if there are at least 4 non-NaN values, and returns NaN otherwise. This is to ensure 
            having NaNs on days with incomplete data when grouping into daily data rather than incorrect data.
            '''
            if np.count_nonzero(~np.isnan(a)) < 4: #incomplete data
                return np.nan
            else:
                return np.max(a)
            
        aggregate[datename] = day
        aggregate['air_temp_min'] = np.min
        aggregate['air_temp_max'] = np.max
        aggregate['air_temp_low_6_hour'] = min_or_nan
        aggregate['air_temp_high_6_hour'] = max_or_nan
        aggregate['precip_accum_one_hour'] = np.sum
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
        obs_hourly_copy = obs_hourly.copy()
        obs_hourly_copy.set_index(datename,inplace=True)
        
        if config['verbose']:
            print('verification: checking matching dates for daily obs and CF6')
        if use_climo:
            try:
                climo_values = _climo(config, station_id, dates)
            except BaseException as e:
                if config['verbose']:
                    print("verification: warning: '%s' while reading climo data" % str(e))
                climo_values = {}
        else:
            if config['verbose']:
                print('verification: not using climo.')
            climo_values = {}
        if use_cf6:
            num_months = min((datetime.utcnow() - dates[0]).days / 30, 24)
            try:
                get_cf6_files(config, station_id, num_months)
            except BaseException as e:
                if config['verbose']:
                    print("verification: warning: '%s' while getting CF6 files" % str(e))
            try:
                cf6_values = _cf6(config, station_id)
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
            use_cf6_precip = False
            if 'air_temp_high_6_hour' in vars_request:
                max_temp_var = 'air_temp_high_6_hour'
            else:
                max_temp_var = 'air_temp_max'  
                    
            if 'air_temp_low_6_hour' in vars_request:
                min_temp_var = 'air_temp_low_6_hour'
            else:
                min_temp_var = 'air_temp_min' 
                    
            if 'precip_accum_six_hour' in vars_request:
                precip_var = 'precip_accum_six_hour'
            else:
                precip_var = 'precip_accum_one_hour'
                
            obs_max_temp = row[max_temp_var]
            obs_min_temp = row[min_temp_var]
            obs_wind = row['wind_speed']
            obs_precip = round(row[precip_var],2)
            if np.isnan(obs_max_temp) and np.isnan(obs_min_temp): #if high or low temperature is missing, chances are some precipitation data is missing too
                use_cf6_precip = True
            
            # Check for missing or incorrect 6-hour precipitation amounts. If there are any, use sum of 1-hour precipitation amounts if none are missing.
            skip_date = False
            if 'precip_accum_six_hour' in vars_request: #6-hour precipitation amounts were used
                daily_precip = 0.0
                for hour in [5,11,17,23]: #check the 4 times which should have 6-hour precipitation amounts
                    try:
                        obs_6hr_precip = round(obs_hourly_copy['precip_accum_six_hour'][pd.Timestamp(date.year,date.month,date.day,hour)],2)
                    except KeyError: #incomplete data for date
                        skip_date = True
                        use_cf6_precip = True
                        break
                    if np.isnan(obs_6hr_precip):
                        obs_6hr_precip = 0.0
                    sum_hourly_precip = 0.0
                    for hour2 in range(hour-5,hour+1): #check and sum 1-hour precipitation amounts
                        obs_hourly_precip = obs_hourly_copy['precip_accum_one_hour'][pd.Timestamp(date.year,date.month,date.day,hour2)]
                        if np.isnan(obs_hourly_precip): #missing 1-hour precipitation amount, so use cf6/climo value instead
                            use_cf6_precip = True
                        else:
                            sum_hourly_precip += round(obs_hourly_precip,2)
                    if sum_hourly_precip > obs_6hr_precip and not use_cf6_precip: #Missing or incorrect 6-hour precipitation amount but 1-hour precipitation amounts are OK
                        obs_6hr_precip = round(sum_hourly_precip,2)
                    daily_precip += round(obs_6hr_precip,2)
                if (round(daily_precip,2) > round(obs_precip,2) and not use_cf6_precip):
                    print('verification: warning: incorrect obs precip of %0.2f for %s, using summed one hour accumulation value of %0.2f' % (obs_precip,date,daily_precip))
                    obs_daily.loc[index, 'precip_accum_six_hour'] = daily_precip
            else: #1-hour precipitation amounts were used
                for hour in range(24):
                    try:
                        obs_hourly_precip = obs_hourly_copy['precip_accum_one_hour'][pd.Timestamp(date.year,date.month,date.day,hour)]
                    except KeyError: #incomplete data for date
                        skip_date = True
                        break
                    if np.isnan(obs_hourly_precip):
                        use_cf6_precip = True
            if skip_date:
                obs_daily.loc[index,max_temp_var] = np.nan
                obs_daily.loc[index,min_temp_var] = np.nan
                obs_daily.loc[index,'wind_speed'] = np.nan
                obs_daily.loc[index,precip_var] = np.nan
            if date in climo_values.keys() and not skip_date:
                count_rows += 1
                cf6_max_temp = climo_values[date]['max_temp']
                cf6_min_temp = climo_values[date]['min_temp']
                cf6_wind = climo_values[date]['wind']
                cf6_precip = climo_values[date]['precip']
                if not (np.isnan(cf6_max_temp)) and cf6_max_temp > -900.0 and np.isnan(obs_max_temp):
                    print('verification: warning: missing obs max temp for %s, using cf6/climo value of %d' % (date,round(cf6_max_temp,0)))
                    obs_daily.loc[index, max_temp_var] = cf6_max_temp
                if not (np.isnan(cf6_min_temp)) and cf6_min_temp < 900.0 and np.isnan(obs_min_temp):
                    print('verification: warning: missing obs min temp for %s, using cf6/climo value of %d' % (date,round(cf6_min_temp,0)))
                    obs_daily.loc[index, min_temp_var] = cf6_min_temp
                if not (np.isnan(cf6_wind)):
                    if obs_wind > cf6_wind and obs_wind < cf6_wind + 10:
                        print('verification: warning: obs wind for %s larger than wind from cf6/climo; using obs' % 
                              date)
                    else:
                        obs_daily.loc[index, 'wind_speed'] = cf6_wind
                else:
                    count_rows -= 1
                if not (np.isnan(cf6_precip)) and cf6_precip > -900.0 and use_cf6_precip and round(cf6_precip,2) > round(obs_precip,2):
                    print('verification: warning: incorrect obs precip of %0.2f for %s, using cf6/climo value of %0.2f' % (obs_precip,date,cf6_precip))
                    obs_daily.loc[index, precip_var] = cf6_precip
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
    
        # Set the date time index and retain only desired columns
        obs_daily = obs_daily.rename(columns={datename: 'date_time'})
        obs_daily = obs_daily.set_index('date_time')
        if config['verbose']:
            print('verification: -> exporting to %s' % output_file)
        export_cols = ['Tmax', 'Tmin', 'Wind', 'Rain']
        for col in obs_daily.columns:
            if col not in export_cols:
                obs_daily.drop(col, 1, inplace=True)
    
        # If a time series is desired, then get hourly data from csv file
        if config['Model']['predict_timeseries']:
            obs_hourly_verify = all_obspd[['date_time', 'air_temp', 'relative_humidity', 'wind_speed', 'precip_accum_one_hour']] #subset of data used as verification
    
            # Fix rainfall for categorical and time accumulation
            rain_column = 'precip_last_%d_hour' % config['time_series_interval']
            obs_hourly_verify.rename(columns={'precip_accum_one_hour': rain_column}, inplace=True)
            if config['Model']['rain_forecast_type'] == 'pop' and not force_rain_quantity:
                if config['verbose']:
                    print("verification: using 'pop' rain")
                obs_hourly_verify.loc[:, rain_column] = pop_rain(obs_hourly_verify[rain_column])
                use_rain_max = True
            elif config['Model']['rain_forecast_type'] == 'categorical' and not force_rain_quantity:
                if config['verbose']:
                    print("verification: using 'categorical' rain")
                obs_hourly_verify.loc[:, rain_column] = categorical_rain(obs_hourly_verify[rain_column])
                use_rain_max = True
            else:
                use_rain_max = False
    
        # Export final data
        export_dict = OrderedDict()
        for date in dates:
            try:
                day_dict = obs_daily.loc[date].to_dict(into=OrderedDict)
            except KeyError:
                continue
            if np.any(np.isnan(list(day_dict.values()))):
                if config['verbose']:
                    print('verification: warning: omitting day %s; missing data' % date)
                continue  # No verification can have missing values
            if config['Model']['predict_timeseries']:
                start = pd.Timestamp(date + timedelta(hours=(config['forecast_hour_start'] -
                                                             config['time_series_interval'])))
                end = pd.Timestamp(date + timedelta(hours=config['forecast_hour_start'] + 24))
                try:
                    series = reindex_hourly(obs_hourly_verify, start, end, config['time_series_interval'],
                                            use_rain_max=use_rain_max)
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
            pickle.dump(export_dict, handle, protocol=2)

    return


def process(config, verif_list):
    """
    Returns a numpy array of verification data for use in mosx_predictors. The first dimension is date, the second is
    variable.
    :param config:
    :param verif: dict: dictionary of processed verification data; may be None
    :return: ndarray: array of processed verification targets
    """
    verif_arrays = []
    if verif_list is not None:
        for i in range(len(verif_list)):
            verif = verif_list[i]
            print('verification.process: processing array for verification data')
            num_days = len(verif.keys())
            variables = ['Tmax', 'Tmin', 'Wind', 'Rain']
            day_verif_array = np.full((num_days, len(variables)), np.nan, dtype=np.float64)
            for d in range(len(verif.keys())):
                date = list(verif.keys())[d]
                for v in range(len(variables)):
                    var = variables[v]
                    day_verif_array[d, v] = verif[date][var]
            if config['Model']['predict_timeseries']:
                hour_verif = OrderedDict(verif)
                for date in hour_verif.keys():
                    for variable in variables:
                        hour_verif[date].pop(variable, None)
                hour_verif_array = get_array(hour_verif)
                hour_verif_array = np.reshape(hour_verif_array, (num_days, -1))
                verif_array = np.concatenate((day_verif_array, hour_verif_array), axis=1)
            else:
                verif_array = day_verif_array
            verif_arrays.append(verif_array)
        return verif_arrays
    else:
        return None
