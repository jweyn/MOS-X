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
import pandas as pd


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
        rain_array[pd.isnull(rain_array)] = 0.

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
    bufr, obs, verif = unpickle(bufr_file, obs_file, verif_file)
    bufr, obs, verif, all_dates = find_matching_dates(bufr, obs, verif, return_data=True)
    bufr_array = mosx.bufr.process(config, bufr)
    obs_array = mosx.obs.process(config, obs)
    verif_array = mosx.verification.process(config, verif)

    export_dict = {
        'BUFKIT': bufr_array,
        'OBS': obs_array,
        'VERIF': verif_array
    }
    export_dict = PredictorDict(export_dict)

    # Get raw precipitation values and add them to the PredictorDict
    precip_list = []
    for date in all_dates:
        precip = []
        items = list(bufr.items())
        for item in items:
            if item[0] == b'DAY' or item[0] == 'DAY':
                bufr_day = item[1]
        for model in bufr_day.keys():
            try:
                precip.append(bufr_day[model][date][-1] / 25.4)  # mm to inches
            except KeyError: #date doesn't exist
                pass
        precip_list.append(precip)
    export_dict.add_rain(np.array(precip_list))

    if output_file is None:
        output_file = '%s/%s_predictors.pkl' % (config['site_directory'], config['station_id'])
    print('predictors: -> exporting to %s' % output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(export_dict, handle, protocol=2)

    if return_dates:
        return all_dates
    else:
        return
        
