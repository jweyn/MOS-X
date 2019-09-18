# MOS-X (modified version of Jonathan Weyn's original repository)

MOS-X is a machine learning-based forecasting model built in Python designed to produce output tailored for the [WxChallenge](http://www.wxchallenge.com) weather forecasting competition.
It uses an external executable to download and process time-height profiles of model data from the National Centers for Environmental Prediction (NCEP) Global Forecast System (GFS) and North American Mesoscale (NAM) models.
These data, along with surface observations from MesoWest, are used to train any of scikit-learn's ML algorithms to predict tomorrow's high temperature, low temperature, peak 2-minute sustained wind speed, and rain total.

## Installing

### Requirements

- Python 2 or 3 (for now I get an unpickling error when predicting using Python 3; use with Python 3 with caution and feel free to raise issues.)
- A workstation with a recent Linux installation... sorry, that's all that will work with the next item...
- [BUFRgruven](http://strc.comet.ucar.edu/software/bgruven/) - for model data
- An API key for [MesoWest](https://synopticlabs.org/api/mesonet/) - unfortunately the API now has a limited free tier. MOS-X currently does a poor job of data caching so large data sets will exceed the free limit - use with caution.
- A decent amount of free disk space - some of the models are > 1 GB pickle files... not to mention all the BUFKIT files...

### Python packages - easier with conda

- NumPy (>= 1.14.0)
- scipy
- pandas (tested on version 0.24.x)
- ConfigObj
- ulmo (use conda-forge)
- the excellent [scikit-learn](http://scikit-learn.org/stable/index.html)
- metpy (tested on version 0.7.0; don't use version 0.8 and up because get_upper_air_data no longer exists)
- pint (0.8.1; pint 0.9 will be automatically installed with metpy but there is a bug in it that will cause errors)

### Installation

Nothing to do really. Just make sure the scripts in the main directory (`build`, `run`, `verify_py2` or `verify_py3`, `validate_py2` or `validate_py3`, and `performance`) are executable, for example:

`chmod +x build run verify_py3 validate_py3 performance`

The main scripts with "py2" or "py3" in their names indicate they are the separate Python 2 and Python 3 versions respectively.

## Building a model

1. The first thing to do is to set up the config file for the particular site to forecast for. The `default.config` file has a good number of comments to describe how to do that. Parameters that are not marked 'optional' or with a default value must be specified.
  - The parameter `climo_station_id` is now automatically generated!
  - It is not recommended to use the upper-air sounding data option. In my testing adding sounding data actually made no difference to the skill of the models, but YMMV. Use with caution. I don't test it. Note: this does not work in Python 3 for now.
2. Once the config is set up, build the model using `build <config>`. The config reader will automatically look for `<config>.config` too, so if you're like me and like to call your config files `KSEA.config`, it's handy to just pass `KSEA`.
  - Depending on how much training data is requested, it may take several hours for BUFRgruven to download everything.
  - Actually building the scikit-learn model, however, takes only 10 minutes for a 1000-tree random forest on a 16-core machine.

## Running the model

- Run the model for tomorrow with `run <config>`, or give it any day to run on.
- Verify the model prediction with the truth and with GFS and NAM MOS products with `verify <config>`.
- The `validate` script basically is a glorified verification over an entire user-specified range of dates.

## Some notes on advanced model configurations

- There is built-in functionality for building a model that predicts a time series of hourly temperature, relative humidity, wind speed, and rain for the forecast period in addition to the daily values. While handy to get an idea of the temporal variation of predicted weather, it actually has limited use, and makes the pickled model file much larger.
- Rain forecasting is difficult for an ML model. Rain values are highly non-normally distributed. There is the option to use a post-processor model, which is another random forest, trained on the distribution of output from the base model's trees. It improves rain forecast a little, particularly by doing a better job of predicting 0 on sunny days.
- Rain forecasting can now be done in three different ways: `quantity`, which is the standard prediction of an actual daily rain total, `pop`, or the probability of precipitation, and `categorical`, which uses the MOS categories.
