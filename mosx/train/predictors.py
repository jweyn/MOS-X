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


def format_predictors(config, bufr_file, obs_file, verif_file, output_file=None, return_dates=False,
                      return_precip_forecast=False):
    """
    Generates a complete date-by-x array of data for ingestion into the machine learning estimator. verif_file may be
    None if creating a set to run the model.

    :param config:
    :param bufr_file: str: full path to the saved file of BUFR data
    :param obs_file: str: full path to the saved file of OBS data
    :param verif_file: str: full path to the saved file of VERIF data
    :param output_file: str: full path to output predictors file
    :param return_dates: if True, returns all of the matching dates used to produce the predictor arrays
    :param return_precip_forecast: if True, returns the raw precipitation data from BUFR models, to compare to MOS-X
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

    if output_file is None:
        output_file = '%s/%s_predictors.pkl' % (config['site_directory'], config['station_id'])
    print('-> Exporting to %s' % output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(export_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_precip_forecast:
        precip_list = []
        for date in all_dates:
            precip = []
            for model in bufr['DAY'].keys():
                precip.append(bufr['DAY'][model][date][-1] / 25.4)  # mm to inches
            precip_list.append(precip)

    if return_dates and return_precip_forecast:
        return all_dates, precip_list
    elif return_dates:
        return all_dates
    elif return_precip_forecast:
        return precip_list
    else:
        return
