#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for processing input predictor data.
"""

import numpy as np
import mosx.bufr
import mosx.obs
import mosx.verification
from mosx.util import unpickle, find_matching_dates
import pickle


class PredictorDict(dict):
    """
    Special class extending dict to add an attribute containing raw precipitation values, for precipitation-aware
    estimator configurations.
    """
    def __init__(self, *args, **kwargs):
        super(PredictorDict, self).__init__(*args, **kwargs)
        self.rain = None

    def add_rain(self, rain_array):
        """
        Add an array of raw rain values to the dict. If the dictionary contains BUFKIT array, checks that the sample
        size is correct.
        :param rain_array:
        :return:
        """
        rain_array[np.isnan(rain_array)] = 0.

        if 'BUFKIT' in self.keys():
            if isinstance(self['BUFKIT'], np.ndarray):
                if self['BUFKIT'].shape[0] != rain_array.shape[0]:
                    raise ValueError('rain_array and BUFKIT array must have the same sample size; got %s and %s' %
                                     (rain_array.shape[0], self['BUFKIT'].shape[0]))
        self.rain = rain_array


def format_predictors(config, bufr_file, obs_file, verif_file, output_file=None, return_dates=False):
    """
    Generates a complete date-by-x array of data for ingestion into the machine learning estimator. verif_file may be
    None if creating a set to run the model.

    :param config:
    :param bufr_file: str: full path to the saved file of BUFR data
    :param obs_file: str: full path to the saved file of OBS data
    :param verif_file: str: full path to the saved file of VERIF data
    :param output_file: str: full path to output predictors file
    :param return_dates: if True, returns all of the matching dates used to produce the predictor arrays
    :return: optionally a list of dates and a list of lists of precipitation values
    """
    print("PREDICTORS!")
    bufr, obs, verif = unpickle(bufr_file, obs_file, verif_file)

    # make a gigantic dataframe of the obs, bufr, and verif predictors
    if verif is not None:
        # can change columns of verification based on config file if more things are being predicted
        verif_df = pd.DataFrame.from_dict(verif, orient='index')

    # create obs dataframe
    obs_dates = [key for key in obs['SFC'].keys()]
    obs_variables = [key for key in obs['SFC'][obs_dates[0]].keys()]
    timestamps = [key for key in obs['SFC'][obs_dates[0]][obs_variables[0]].keys()]
    hhmm_stamps = [str(timestamp.hour).zfill(2)+str(timestamp.minute)+'Z' for timestamp in timestamps]

    obs_columns = []
    for var in obs_variables:
        for hhmm in hhmm_stamps:
            obs_columns.append('{} {}'.format(var, hhmm))
    obs_df = pd.DataFrame(index=obs_dates, columns=obs_columns)

    # fill obs dataframe
    obs_array = np.squeeze(get_array(obs)).reshape(len(obs_dates), len(obs_columns))
    obs_df[:] = obs_array

    # this is the annoying part where we extract all the keys from the ordered dictionary
    # use index 0 for the first model--multiple models need to match
    bufr_keys = [key for key in bufr[0]['PROF'][fcst_day].keys()]
    timestamps_prof = [key for key in bufr[0]['PROF'][fcst_day][bufr_keys[0]].keys()]
    timestamps_sfc = [key for key in bufr[0]['SFC'][fcst_day][bufr_keys[0]].keys()]
    # hhmm_stamps are the hours since the start of the forecast period (0-24 hr)
    hhmm_stamps_prof = np.linspace(0, 24, len(timestamps_prof)).astype('int').astype('str')
    hhmm_stamps_sfc = np.linspace(0, 24, len(timestamps_sfc)).astype('int').astype('str')
    fcst_key_tmp = [key for key in bufr[0]['PROF'][fcst_day][bufr_keys[0]].keys()][0]
    var_keys_prof = [key for key in bufr[0]['PROF'][fcst_day][bufr_keys[0]][fcst_key_tmp].keys()]
    var_keys_sfc = [key for key in bufr[0]['SFC'][fcst_day][bufr_keys[0]][fcst_key_tmp].keys()]

    pressure_levels = [int(i) for i in config['Bufkit']['pressure_levels']]
    columns_nomodel_prof = []
    columns_nomodel_sfc = []
    columns_nomodel_day = ['high', 'low', 'max 10m wind', 'precip']
    for var in var_keys_prof:
        for pres in pressure_levels:
            for hm in hhmm_stamps_prof:
                columns_nomodel_prof.append('{} {} {}'.format(var, pres, hm))
    for var in var_keys_sfc:
        for hm in hhmm_stamps_sfc:
            columns_nomodel_sfc.append('{} {}'.format(var, hm))

    # fill bufr dataframe
    bufr_df = pd.DataFrame(index=bufr_keys)
    for i, model in enumerate(models):
        columns_prof = ['{} {}'.format(model, column) for column in columns_nomodel_prof]
        columns_sfc = ['{} {}'.format(model, column) for column in columns_nomodel_sfc]
        columns_day = ['{} {}'.format(model, column) for column in columns_nomodel_day]
        prof_df = pd.DataFrame(index=bufr_keys, columns=columns_prof)
        prof_array = np.squeeze(get_array(bufr[0]['PROF'])).reshape(len(bufr_keys), len(columns_prof))
        prof_df[:] = prof_array
        bufr_df = pd.concat((bufr_df, prof_df), axis=1)
        sfc_df = pd.DataFrame(index=bufr_keys, columns=columns_sfc)
        sfc_array = np.squeeze(get_array(bufr[0]['SFC'])).reshape(len(bufr_keys), len(columns_sfc))
        sfc_df[:] = sfc_array
        bufr_df = pd.concat((bufr_df, sfc_df), axis=1)
        day_df = pd.DataFrame(index=bufr_keys, columns=columns_day)
        day_array = np.squeeze(get_array(bufr[0]['DAY'])).reshape(len(bufr_keys), len(columns_day))
        day_df[:] = day_array
        bufr_df = pd.concat((bufr_df, day_df), axis=1)

    if verif is not None:
        final_index = bufr_df.index.intersection(obs_df.index).intersection(verif_df.index)
    else:
        final_index = bufr_df.index.intersection(obs_df.index)

    # finally trim the bufr, obs, and verif dataframes with the final index
    bufr_df = bufr_df.loc[final_index]
    obs_df = obs_df.loc[final_index]
    if verif is not None:
        verif_df = verif_df.loc[final_index]
    else:
        verif_df = None

    export_dict = {
        'BUFKIT': bufr_df,
        'OBS': obs_df,
        'VERIF': verif_df
    }
    export_dict = PredictorDict(export_dict)

    # Get raw precipitation values and add them to the PredictorDict
    precip_list = []
    for date in bufr_df.index:
        precip = []
        for model in bufr['DAY'].keys():
            precip.append(bufr['DAY'][model][date][-1] / 25.4)  # mm to inches
        precip_list.append(precip)
    export_dict.add_rain(np.array(precip_list))

    if output_file is None:
        output_file = '%s/%s_predictors.pkl' % (config['site_directory'], config['station_id'])
    print('predictors: -> exporting to %s' % output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(export_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_dates:
        return bufr_df.index
    else:
        return
