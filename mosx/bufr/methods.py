#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for processing BUFR data.
"""

from mosx.util import generate_dates, get_array
from collections import OrderedDict
import re
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from scipy import interpolate


def bufkit_parser_time_height(config, file_name, interval=1, start_dt=None, end_dt=None):
    """
    By Luke Madaus. Modified by jweyn.
    Returns a dictionary of time-height profiles from a BUFKIT file, with profiles interpolated to a basic set of
    pressure levels.
    :param config:
    :param file_name: str: full path to bufkit file name
    :param interval: int: process data every 'interval' hours
    :param start_dt: datetime: starting time for data processing
    :param end_dt: datetime: ending time for data processing
    :return: dict: ugly dictionary of processed values
    """
    # Open the file
    infile = open(file_name, 'r')

    profile = OrderedDict()

    # Find the block that contains the description of what everything is (header information)
    block_lines = []
    inblock = False
    block_found = False
    for line in infile:
        if line.startswith('PRES TMPC') and not block_found:
            # We've found the line that starts the header info
            inblock = True
            block_lines.append(line)
        elif inblock:
            # Keep appending lines until we start hitting numbers
            if re.match('^\d{3}|^\d{4}', line):
                inblock = False
                block_found = True
            else:
                block_lines.append(line)

    # Now compute the remaining number of variables
    re_string = ''
    for line in block_lines:
        dum_num = len(line.split())
        for n in range(dum_num):
            re_string = re_string + '(-?\d{1,5}.\d{2}) '
        re_string = re_string[:-1]  # Get rid of the trailing space
        re_string = re_string + '\r\n'

    # Compile this re_string for more efficient re searches
    block_expr = re.compile(re_string)

    # Now get corresponding indices of the variables we need
    full_line = ''
    for r in block_lines:
        full_line = full_line + r[:-2] + ' '
    # Now split it
    varlist = re.split('[ /]', full_line)
    # Get rid of trailing space
    varlist = varlist[:-1]

    # Variables we want
    vars_desired = ['TMPC', 'DWPC', 'UWND', 'VWND', 'HGHT']

    # Pressure levels to interpolate to
    plevs = [600, 750, 850, 925]
    plevs = [p for p in plevs if p <= float(config['lowest_p_level'])]

    # We now need to break everything up into a chunk for each
    # forecast date and time
    with open(file_name) as infile:
        blocks = infile.read().split('STID')
        for block in blocks:
            interp_plevs = []
            header = block
            if header.split()[0] != '=':
                continue
            fcst_time = re.search('TIME = (\d{6}/\d{4})', header).groups()[0]
            fcst_dt = datetime.strptime(fcst_time, '%y%m%d/%H%M')
            if start_dt is not None and fcst_dt < start_dt:
                continue
            if end_dt is not None and fcst_dt > end_dt:
                break
            if fcst_dt.hour % interval != 0:
                continue
            temp_vars = OrderedDict()
            for var in varlist:
                temp_vars[var] = []
            temp_vars['PRES'] = []
            for block_match in block_expr.finditer(block):
                vals = block_match.groups()
                for val, name in zip(vals, varlist):
                    if float(val) == -9999.:
                        temp_vars[name].append(np.nan)
                    else:
                        temp_vars[name].append(float(val))

            # Unfortunately, bufkit values aren't always uniformly distributed.
            final_vars = OrderedDict()
            cur_plevs = temp_vars['PRES']
            cur_plevs.reverse()
            for var in varlist[1:]:
                if var in (vars_desired + ['SKNT', 'DRCT']):
                    values = temp_vars[var]
                    values.reverse()
                    interp_plevs = list(plevs)
                    num_plevs = len(interp_plevs)
                    f = interpolate.interp1d(cur_plevs, values, bounds_error=False)
                    interp_vals = f(interp_plevs)
                    interp_array = np.full((len(plevs)), np.nan)
                    # Array almost certainly missing values at high pressures
                    interp_array[:num_plevs] = interp_vals
                    interp_vals = list(interp_array)
                    interp_plevs = list(plevs)  # use original array
                    interp_vals.reverse()
                    interp_plevs.reverse()
                    if var == 'SKNT':
                        wspd = np.array(interp_vals)
                    if var == 'DRCT':
                        wdir = np.array(interp_vals)
                if var in vars_desired:
                    final_vars[var] = interp_vals
            final_vars['PRES'] = interp_plevs
            if 'UWND' not in final_vars.keys():
                final_vars['UWND'] = list(wspd * np.sin(wdir * np.pi/180. - np.pi))
            if 'VWND' not in final_vars.keys():
                final_vars['VWND'] = list(wspd * np.cos(wdir * np.pi/180. - np.pi))
            profile[fcst_dt] = final_vars

    return profile


def bufkit_parser_surface(file_name, interval=1, start_dt=None, end_dt=None):
    """
    By Luke Madaus. Modified by jweyn.
    Returns a dictionary of surface data from a BUFKIT file.
    :param file_name: str: full path to bufkit file name
    :param interval: int: process data every 'interval' hours
    :param start_dt: datetime: starting time for data processing
    :param end_dt: datetime: ending time for data processing
    :return: dict: ugly dictionary of processed values
    """
    # Load the file
    infile = open(file_name, 'r')
    sfc_dict = OrderedDict()

    block_lines = []
    inblock = False
    for line in infile:
        if re.search('SELV', line):
            try:  # jweyn
                elev = re.search('SELV = -?(\d{1,4})', line).groups()[0]  # jweyn: -?
                elev = float(elev)
            except:
                elev = 0.0
        if line.startswith('STN YY'):
            # We've found the line that starts the header info
            inblock = True
            block_lines.append(line)
        elif inblock:
            # Keep appending lines until we start hitting numbers
            if re.search('\d{6}', line):
                inblock = False
            else:
                block_lines.append(line)

    # Build an re search pattern based on this
    # We know the first two parts of the section are station id num and date
    re_string = "(\d{6}|\w{4}) (\d{6})/(\d{4})"
    # Now compute the remaining number of variables
    dum_num = len(block_lines[0].split()) - 2
    for n in range(dum_num):
        re_string = re_string + " (-?\d{1,4}.\d{2})"
    re_string = re_string + '\r\n'
    for line in block_lines[1:]:
        dum_num = len(line.split())
        for n in range(dum_num):
            re_string = re_string + '(-?\d{1,4}.\d{2}) '
        re_string = re_string[:-1]  # Get rid of the trailing space
        re_string = re_string + '\r\n'

    # Compile this re_string for more efficient re searches
    block_expr = re.compile(re_string)

    # Now get corresponding indices of the variables we need
    full_line = ''
    for r in block_lines:
        full_line = full_line + r[:-2] + ' '
    # Now split it
    varlist = re.split('[ /]', full_line)

    with open(file_name) as infile:
        # Now loop through all blocks that match the search pattern we defined above
        blocknum = -1
        # For rain total in missing times
        temp_rain = 0.
        # For max temp, min temp, max wind, total rain
        t_max = -150.
        t_min = 150.
        w_max = 0.
        r_total = 0.
        for block_match in block_expr.finditer(infile.read()):
            blocknum += 1
            # Split out the match into each component number
            vals = block_match.groups()
            # Check for missing values
            for v in range(len(vals)):
                if vals[v] == -9999.:
                    vals[v] = np.nan
            # Set the time
            dt = '20' + vals[varlist.index('YYMMDD')] + vals[varlist.index('HHMM')]
            validtime = datetime.strptime(dt, '%Y%m%d%H%M')

            # Check that time is within the period we want
            if start_dt is not None and validtime < start_dt:
                continue
            if end_dt is not None and validtime > end_dt:
                break

            # Check for max daily values!
            t_max = max(t_max, float(vals[varlist.index('T2MS')]))
            t_min = min(t_min, float(vals[varlist.index('T2MS')]))
            uwind = float(vals[varlist.index('UWND')])
            vwind = float(vals[varlist.index('VWND')])
            wspd = np.sqrt(uwind ** 2 + vwind ** 2)
            w_max = max(w_max, wspd)

            if validtime.hour % interval != 0:
                # Still need to get cumulative precipitation!
                try:
                    temp_precip = float(vals[varlist.index('P01M')])
                except:
                    temp_precip = float(vals[varlist.index('P03M')])
                if np.isnan(temp_precip):
                    temp_precip = 0.0
                temp_rain += temp_precip
                continue

            sfc_dict[validtime] = OrderedDict()
            sfc_dict[validtime]['WSPD'] = wspd
            sfc_dict[validtime]['UWND'] = uwind
            sfc_dict[validtime]['VWND'] = vwind
            sfc_dict[validtime]['PRES'] = float(vals[varlist.index('PRES')])
            sfc_dict[validtime]['TMPC'] = float(vals[varlist.index('T2MS')])
            sfc_dict[validtime]['DWPC'] = float(vals[varlist.index('TD2M')])
            sfc_dict[validtime]['HCLD'] = float(vals[varlist.index('HCLD')])
            sfc_dict[validtime]['MCLD'] = float(vals[varlist.index('MCLD')])
            sfc_dict[validtime]['LCLD'] = float(vals[varlist.index('LCLD')])
            # Could be 3 hour or 1 hour precip
            try:
                precip = float(vals[varlist.index('P01M')])
            except:
                precip = float(vals[varlist.index('P03M')])
            # Make sure precip is not nan
            if np.isnan(precip):
                precip = 0.0
            # Add the temporary value from uneven intervals
            precip += temp_rain
            # Also do cumulative sum in r_total
            previous_time = validtime - timedelta(hours=interval)
            if previous_time in sfc_dict.keys():
                sfc_dict[validtime]['PRCP'] = precip
                r_total += precip
            else:
                # We want zero at first time: precip in LAST hour or three
                sfc_dict[validtime]['PRCP'] = 0.0
            # Reset temp_rain after having added it
            temp_rain = 0.

    daily = [t_max, t_min, w_max, r_total]
    return sfc_dict, daily


def bufr_retrieve(bufr, bufarg):
    """
    Call bufrgruven to retrieve BUFR files.
    :param bufr: str: bufrgruven executable path
    :param bufarg: dict: dictionary of arguments passed to bufrgruven
    :return:
    """
    argstring = ''
    for key, value in bufarg.items():
        argstring += ' --%s %s' % (key, value)
    result = os.system('%s %s' % (bufr, argstring))
    return result


def bufr(config, output_file=None, cycle='18'):
    """
    Generates model data from BUFKIT profiles and saves to a file, which can later be retrieved for either training
    data or model run data.
    :param config:
    :param output_file: str: output file path
    :param cycle: str: model cycle (init hour)
    :return:
    """
    bufr_station_id = config['BUFR']['bufr_station_id']
    # Base arguments dictionary. dset and date will be modified iteratively.
    bufarg = {
        'dset': '',
        'date': '',
        'cycle': cycle,
        'stations': bufr_station_id.lower(),
        'noascii': '',
        'noverbose': '',
        'nozipit': '',
        'prepend': ''
    }
    if config['verbose']:
        print('\n')
    bufr_default_dir = '%s/metdat/bufkit' % config['BUFR_ROOT']
    bufr_data_dir = config['BUFR']['bufr_data_dir']
    if not(os.path.isdir(bufr_data_dir)):
        os.makedirs(bufr_data_dir)
    bufrgruven = config['BUFR']['bufrgruven']
    if config['verbose']:
        print('bufr: using BUFKIT files in %s' % bufr_data_dir)
    bufr_format = '%s/%s%s.%s_%s.buf'
    missing_dates = []
    models = config['BUFR']['bufr_models']
    model_names = config['BUFR']['models']
    start_date = datetime.strptime(config['data_start_date'], '%Y%m%d') - timedelta(days=1)
    end_date = datetime.strptime(config['data_end_date'], '%Y%m%d') - timedelta(days=1)
    dates = generate_dates(config, start_date=start_date, end_date=end_date)
    for date in dates:
        bufarg['date'] = datetime.strftime(date, '%Y%m%d')
        if date.year < 2010:
            if config['verbose']:
                print('bufr: skipping BUFR data for %s; data starts in 2010.' % bufarg['date'])
            continue
        if config['verbose']:
            print('bufr: date: %s' % bufarg['date'])

        for m in range(len(models)):
            if config['verbose']:
                print('bufr: trying to retrieve BUFR data for %s...' % model_names[m])
            bufr_new_name = bufr_format % (bufr_data_dir, bufarg['date'], '%02d' % int(bufarg['cycle']),
                                           model_names[m], bufarg['stations'])
            if os.path.isfile(bufr_new_name):
                if config['verbose']:
                    print('bufr: file %s already exists; skipping!' % bufr_new_name)
                break

            if type(models[m]) == list:
                for model in models[m]:
                    try:
                        bufarg['dset'] = model
                        bufr_retrieve(bufrgruven, bufarg)
                        bufr_name = bufr_format % (bufr_default_dir, bufarg['date'], '%02d' % int(bufarg['cycle']),
                                                   model, bufarg['stations'])
                        bufr_file = open(bufr_name)
                        bufr_file.close()
                        os.rename(bufr_name, bufr_new_name)
                        if config['verbose']:
                            print('bufr: BUFR file found for %s at date %s.' % (model, bufarg['date']))
                            print('bufr: writing BUFR file: %s' % bufr_new_name)
                        break
                    except:
                        if config['verbose']:
                            print('bufr: BUFR file for %s at date %s not retrieved.' % (model, bufarg['date']))
            else:
                try:
                    model = models[m]
                    bufarg['dset'] = model
                    bufr_retrieve(bufrgruven, bufarg)
                    bufr_name = bufr_format % (bufr_default_dir, bufarg['date'], '%02d' % int(bufarg['cycle']),
                                               bufarg['dset'], bufarg['stations'])
                    bufr_file = open(bufr_name)
                    bufr_file.close()
                    os.rename(bufr_name, bufr_new_name)
                    if config['verbose']:
                        print('bufr: BUFR file found for %s at date %s.' % (model, bufarg['date']))
                        print('bufr: writing BUFR file: %s' % bufr_new_name)
                except:
                    if config['verbose']:
                        print('bufr: BUFR file for %s at date %s not retrieved.' % (model, bufarg['date']))
            if not (os.path.isfile(bufr_new_name)):
                print('bufr: warning: no BUFR file found for model %s at date %s' % (
                    model_names[m], bufarg['date']))
                missing_dates.append((date, model_names[m]))

    # Process data
    print('\n')
    bufr_dict = OrderedDict({'PROF': OrderedDict(), 'SFC': OrderedDict(), 'DAY': OrderedDict()})
    for model in model_names:
        bufr_dict['PROF'][model] = OrderedDict()
        bufr_dict['SFC'][model] = OrderedDict()
        bufr_dict['DAY'][model] = OrderedDict()

    for date in dates:
        date_str = datetime.strftime(date, '%Y%m%d')
        verif_date = date + timedelta(days=1)
        start_dt = verif_date + timedelta(hours=config['forecast_hour_start'])
        end_dt = verif_date + timedelta(hours=config['forecast_hour_start'] + 24)
        for model in model_names:
            if (date, model) in missing_dates:
                if config['verbose']:
                    print('bufr: skipping %s data for %s; file missing.' % (model, date_str))
                continue
            if config['verbose']:
                print('bufr: processing %s data for %s' % (model, date_str))
            bufr_name = bufr_format % (bufr_data_dir, date_str, '%02d' % int(bufarg['cycle']), model,
                                       bufarg['stations'])
            if not (os.path.isfile(bufr_name)):
                if config['verbose']:
                    print('bufr: skipping %s data for %s; file missing.' % (model, date_str))
                continue
            profile = bufkit_parser_time_height(config, bufr_name, 6, start_dt, end_dt)
            sfc, daily = bufkit_parser_surface(bufr_name, 3, start_dt, end_dt)
            # Drop 'PRES' variable which is useless
            for key, values in profile.items():
                values.pop('PRES', None)
                profile[key] = values
            bufr_dict['PROF'][model][verif_date] = profile
            bufr_dict['SFC'][model][verif_date] = sfc
            bufr_dict['DAY'][model][verif_date] = daily

    # Export data
    if output_file is None:
        output_file = '%s/%s_bufr.pkl' % (config['SITE_ROOT'], config['station_id'])
    if config['verbose']:
        print('bufr: -> exporting to %s' % output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(bufr_dict, handle, protocol=2)

    return


def process(config, bufr, advection_diagnostic=True):
    """
    Imports the data contained in a bufr dictionary and returns a time-by-x numpy array for use in mosx_predictors. The
    first dimension is date; all other dimensions are first extracted using get_array and then one-dimensionalized.
    :param config:
    :param bufr: dict: dictionary of processed BUFR data
    :param advection_diagnostic: bool: if True, add temperature advection diagnostic to the data
    :return: ndarray: array of formatted BUFR predictor values
    """
    if config['verbose']:
        print('bufr.process: processing array for BUFR data...')
    # PROF part of the BUFR data
    items = list(bufr.items())
    for item in items: #item = (key, value) pair
        if item[0] == b'PROF': #look for 'BUFR' key
            bufr_prof = item[1]
    bufr_prof = get_array(bufr_prof)
    bufr_dims = list(range(len(bufr_prof.shape)))
    bufr_dims[0] = 1
    bufr_dims[1] = 0
    bufr_prof = bufr_prof.transpose(bufr_dims)
    bufr_shape = bufr_prof.shape
    bufr_reshape = [bufr_shape[0]] + [np.cumprod(bufr_shape[1:])[-1]]
    bufr_prof = bufr_prof.reshape(tuple(bufr_reshape))
    # SFC part of the BUFR data
    for item in items: #item = (key, value) pair
        if item[0] == b'SFC': #look for 'SFC' key
            bufr_sfc = item[1]
    bufr_sfc = get_array(bufr_sfc)
    bufr_dims = list(range(len(bufr_sfc.shape)))
    bufr_dims[0] = 1
    bufr_dims[1] = 0
    bufr_sfc = bufr_sfc.transpose(bufr_dims)
    bufr_shape = bufr_sfc.shape
    bufr_reshape = [bufr_shape[0]] + [np.cumprod(bufr_shape[1:])[-1]]
    bufr_sfc = bufr_sfc.reshape(tuple(bufr_reshape))
    # DAY part of the BUFR data
    for item in items: #item = (key, value) pair
        if item[0] == b'DAY': #look for 'DAY' key
            bufr_day = item[1]
    bufr_day = get_array(bufr_day)
    bufr_dims = list(range(len(bufr_day.shape)))
    bufr_dims[0] = 1
    bufr_dims[1] = 0
    bufr_day = bufr_day.transpose(bufr_dims)
    bufr_shape = bufr_day.shape
    bufr_reshape = [bufr_shape[0]] + [np.cumprod(bufr_shape[1:])[-1]]
    bufr_day = bufr_day.reshape(tuple(bufr_reshape))
    bufr_out = np.concatenate((bufr_prof, bufr_sfc, bufr_day), axis=1)
    # Fix missing values
    bufr_out[bufr_out < -1000.] = np.nan
    if advection_diagnostic:
        advection_array = temp_advection(bufr)
        bufr_out = np.concatenate((bufr_out, advection_array), axis=1)
    return bufr_out


def temp_advection(bufr):
    """
    Produces an array of temperature advection diagnostic from the bufr dictionary. The diagnostic is a simple
    calculation of the strength of backing or veering of winds with height, based on thermal wind approximations,
    between the lowest two profile levels retained in the bufr dictionary. Searches for UWND and VWND keys.
    IMPORTANT: expects the keys of each model to match dates; i.e., expects bufr output from find_matching_dates. This
    is to ensure num_samples is correct.
    :param bufr: dict: dictionary of processed BUFR data
    :return: advection_array: array of num_samples-by-num_features of advection diagnostic
    """
    items = list(bufr.items())
    for item in items: #item = (key, value) tuple
        if item[0] == b'PROF': #look for 'PROF' key
            bufr_prof = item[1]
    models = list(bufr_prof.keys())
    num_models = len(models)
    dates = list(bufr_prof[list(bufr_prof.keys())[0]].keys())
    num_dates = len(dates)
    num_times = len(bufr_prof[list(bufr_prof.keys())[0]][dates[0]].keys())
    num_features = num_models * num_times

    advection_array = np.zeros((num_dates, num_features))

    def advection_index(V1, V2):
        """
        The advection index measures the strength of veering/backing of wind.
        :param V1: array wind vector at lower model level
        :param V2: array wind vector at higher model level
        :return: index of projection of (V2 - V1) onto V1
        """
        proj = V2 - np.dot(V1, V2) * V1 / np.linalg.norm(V1)
        diff = V1 - V2
        sign = np.sign(np.arctan2(diff[1], diff[0]))
        return sign * np.linalg.norm(proj)

    # Here comes the giant ugly loop.
    sample = 0
    for date in dates:
        feature = 0
        for model in models:
            try:
                for eval_date in bufr_prof[model][date].keys():
                    items = bufr_prof[model][date][eval_date].items()
                    for item in items: #item = (key, value) tuple
                        if item[0] == b'UWND': #look for 'UWND' key
                            u = item[1]
                        if item[0] == b'VWND': #look for 'VWND' key
                            v = item[1]
                    try:
                        V1 = np.array([u[0], v[0]])
                        V2 = np.array([u[1], v[1]])
                    except IndexError:
                        print('Not enough wind levels available for advection calculation; omitting...')
                        return
                    advection_array[sample, feature] = advection_index(V1, V2)
                    feature += 1
            except KeyError: #date doesn't exist
                pass
        sample += 1

    return advection_array
