#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
MOS-X python package. Tools for building, running, and validating a machine learning model for predicting tomorrow's
weather. 'MOS-X' represents an intelligent improvement to the tradition Model Output Statistics of the National
Weather Service and other meteorological institutions.

============
Version history
------
2017-10-26: Version 0.0
2017-11-03: Version 0.2
    -- added upper-air sounding data from metpy
2017-11-07: Version 0.3
    -- fixed rain not counted in cumulative values from skipped intervals in bufkit profiles
    -- fixed issue with end year when season overlaps to new year
2017-11-22: Version 0.3.1
    -- added optional directory to save bufkit files
    -- added saving of sounding files
    -- added options for random forest object
    -- edit return_precip option in mosx_predictors to handle multiple days
    -- add return_dates option to mosx_predictors
2017-12-06: Version 0.4
    -- added bufkit model predictions of high, low, and max wind
    -- added option to select neural network regressor
    -- changed verification to ONLY use CF6/climo wind if available
    -- added option to not use existing sounding files in mosx_obs
    -- change upper_air to not save nan soundings
    -- improved handling of missing verification values with missing cf6 wind
    -- added options to ignore cf6 and/or climo in mosx_verif
2018-01-16: Version 0.5
    -- added training design to individually train a forest for each weather parameter
    -- added the option to use any scikit-learn regressor
    -- added use_soundings option to optionally omit sounding data
    -- changed pressure levels to 925, 850, 750, 600
    -- pondered adding model times 18Z and 00Z before forecast start time
    -- changed obs dt to 3 hrs
    -- changed bufkit surface dt to 3 hrs
    -- added methodology for generating diagnostic variables from BUFR
    -- added 'temperature advection index' using above
2018-01-18: Version 0.6
    -- added ability to forecast hourly time series
    -- added class for time series estimator
    -- fixed bug where last 6 hours were not included in obs verifications at the end of a season
    -- improved handling of array conversion of obs data by requiring pandas to export OrderedDict and
       using get_array commands
    -- fixed a bug where daily verification would fail when verification is unavailable for a day
2018-03-06: Version 0.6.1
    -- fixed critical bug where obs dataframe was not sorted when running for only one day
2018-03-12: Version 0.6.2
    -- added support for the Ada Boosting estimator wrapper
2018-03-14: Version 0.7.0
    -- completely changed the structure of the project to be modular
2018-03-20: Version 0.7.1
    -- fixed a bug that would result in an extra set of API dates when is_season is False
    -- fixed some issues related to the use of the config file
2018-03-21: Version 0.8.0
    -- added scorers
    -- added learning curves in performance metrics; more to come
2018-03-27: Version 0.8.1
    -- fixed an error in util.generate_dates that failed to produce all the dates when is_season is False
2018-04-10: Version 0.9.0
    -- better implementation of base estimator attributes in TimeSeriesEstimator and RainTuningEstimator classes
    -- added submodule 'predict' for unified predictions
    -- improved handling of the raw precipitation values to allow them to be used in rain tuning
    -- added config option for the starting hour of a forecast day
    -- added config option to predict probability of precipitation rather than quantity
    -- added automatic fetching of CF6 files
    -- added automatic retrieval of climo_station_id (removed from default.config)
    -- added time_series_interval parameter to output coarser time series
    -- added option to disable climo/cf6 wind retrieval, for non-WxChallenge purposes
2018-05-03: Version 0.10.0
    -- moved special estimator classes to mosx.estimators
    -- added bootstrapping training estimator
    -- added ability to select the type of estimator for rain tuning
    -- re-organized the 'train' and 'predict' modules to 'model'
2018-06-11: Version 0.10.2
    -- fixed an error in obs retrieval that retrieved one data point too many
    -- added many plots to 'performance'
2018-10-22: Version 0.10.3
    -- added option to write more than one file type

"""

from .bufr import *
from .estimators import *
from .obs import *
from .model import *
from .util import *
from .verification import *

__version__ = '0.10.3'
