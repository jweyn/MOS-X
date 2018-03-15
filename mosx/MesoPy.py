# ==================================================================================================================== #
# MesoPy                                                                                                               #
# Version: 2.0.0                                                                                                       #
# Copyright (c) 2015-17 MesoWest Developers <atmos-mesowest@lists.utah.edu>                                            #
#                                                                                                                      #
# LICENSE:                                                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated         #
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the  #
# rights to use,copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to       #
# permit persons to whom the Software is furnished to do so, subject to the following conditions:                      #
#                                                                                                                      #
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the #
# Software.                                                                                                            #
#                                                                                                                      #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE #
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS   #
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,WHETHER IN AN ACTION OF CONTRACT, TORT OR   #
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.     #
#                                                                                                                      #
# ==================================================================================================================== #

try:
    import urllib.parse
    import urllib.request
    import urllib.error
except ImportError:
    import urllib2
    import urllib

import json


# ==================================================================================================================== #
# MesoPyError class                                                                                                    #
# Type: Exception                                                                                                      #
# Description: This class is simply the means for error handling when an exception is raised.                          #
# ==================================================================================================================== #


class MesoPyError(Exception):
    def __init__(self, error_message):
        self.error_message = error_message

    def __str__(self):
        r""" This just returns one of the error messages listed in the checkresponse() function"""
        return repr(self.error_message)


# ==================================================================================================================== #
# Meso class                                                                                                           #
# Type: Main                                                                                                           #
# Description: This class defines an instance of MesoPy and takes in the user's token                                  #
# ==================================================================================================================== #


class Meso(object):
    def __init__(self, token):
        r""" Instantiates an instance of MesoPy.

        Arguments:
        ----------
        token: string, mandatory
            Your API token that authenticates you for requests against MesoWest.mes

        Returns:
        --------
            None.

        Raises:
        -------
            None.
        """

        self.base_url = 'http://api.mesowest.net/v2/'
        self.token = token
        self.geo_criteria = ['stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone', 'gacc',
                             'subgacc']

    # ================================================================================================================ #
    # Functions:                                                                                                       #
    # ================================================================================================================ #

    @staticmethod
    def _checkresponse(response):
        r""" Returns the data requested by the other methods assuming the response from the API is ok. If not, provides
        error handling for all possible API errors. HTTP errors are handled in the get_response() function.

        Arguments:
        ----------
            None.

        Returns:
        --------
            The response from the API as a dictionary if the API code is 2.

        Raises:
        -------
            MesoPyError: Gives different response messages depending on returned code from API. If the response is 2,
            resultsError is displayed. For a response of 200, an authError message is shown. A ruleError is displayed
            if the code is 400, a formatError for -1, and catchError for any other invalid response.

        """

        results_error = 'No results were found matching your query'
        auth_error = 'The token or API key is not valid, please contact Josh Clark at joshua.m.clark@utah.edu to ' \
                     'resolve this'
        rule_error = 'This request violates a rule of the API. Please check the guidelines for formatting a data ' \
                     'request and try again'
        catch_error = 'Something went wrong. Check all your calls and try again'

        if response['SUMMARY']['RESPONSE_CODE'] == 1:
            return response
        elif response['SUMMARY']['RESPONSE_CODE'] == 2:
            raise MesoPyError(results_error)
        elif response['SUMMARY']['RESPONSE_CODE'] == 200:
            raise MesoPyError(auth_error)
        elif response['SUMMARY']['RESPONSE_CODE'] == 400:
            raise MesoPyError(rule_error)
        elif response['SUMMARY']['RESPONSE_CODE'] == -1:
            format_error = response['SUMMARY']['RESPONSE_MESSAGE']
            raise MesoPyError(format_error)
        else:
            raise MesoPyError(catch_error)

    def _get_response(self, endpoint, request_dict):
        """ Returns a dictionary of data requested by each function.

        Arguments:
        ----------
        endpoint: string, mandatory
            Set in all other methods, this is the API endpoint specific to each function.
        request_dict: string, mandatory
            A dictionary of parameters that are formatted into the API call.

        Returns:
        --------
            response: A dictionary that has been dumped from JSON.

        Raises:
        -------
            MesoPyError: Overrides the exceptions given in the requests library to give more custom error messages.
            Connection_error occurs if no internet connection exists. Timeout_error occurs if the request takes too
            long and redirect_error is shown if the url is formatted incorrectly.

        """
        http_error = 'Could not connect to the API. This could be because you have no internet connection, a parameter' \
                     ' was input incorrectly, or the API is currently down. Please try again.'
                     
        json_error = 'Could not retrieve JSON values. Try again with a shorter date range.'
        
        # For python 3.4
        try:
            qsp = urllib.parse.urlencode(request_dict, doseq=True)
            resp = urllib.request.urlopen(self.base_url + endpoint + '?' + qsp).read()

        # For python 2.7
        except AttributeError or NameError:
            try:
                qsp = urllib.urlencode(request_dict, doseq=True)
                resp = urllib2.urlopen(self.base_url + endpoint + '?' + qsp).read()
            except urllib2.URLError:
                raise MesoPyError(http_error)
        except urllib.error.URLError:
            raise MesoPyError(http_error)
        
        try:
            json_data = json.loads(resp.decode('utf-8'))
        except ValueError:
            raise MesoPyError(json_error)
        
        return self._checkresponse(json_data)

    def _check_geo_param(self, arg_list):
        r""" Checks each function call to make sure that the user has provided at least one of the following geographic
        parameters: 'stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone', 'gacc', or 'subgacc'.

        Arguments:
        ----------
        arg_list: list, mandatory
            A list of kwargs from other functions.

        Returns:
        --------
            None.

        Raises:
        -------
            MesoPyError if no geographic search criteria is provided.

        """

        geo_func = lambda a, b: any(i in b for i in a)
        check = geo_func(self.geo_criteria, arg_list)
        if check is False:
            raise MesoPyError('No stations or geographic search criteria specified. Please provide one of the '
                              'following: stid, state, county, country, radius, bbox, cwa, nwsfirezone, gacc, subgacc')

    def attime(self, **kwargs):
        r""" Returns a dictionary of latest observations at a user specified location for a specified time. Users must
        specify at least one geographic search parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa',
        'nwsfirezone', 'gacc', or 'subgacc') to obtain observation data. Other parameters may also be included. See
        below for optional parameters. Also see the metadata() function for station IDs.

        Arguments:
        ----------
        attime: string, required
            Date and time in form of YYYYMMDDhhmm for which returned obs are closest. All times are UTC. e.g.
            attime='201504261800'
        within: string, required
            Can be a single number representing a time period before attime or two comma separated numbers representing
            a period before and after the attime e.g. attime='201306011800', within='30' would return the ob closest to
            attime within a 30 min period before or after attime.
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: string, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        status: string, optional
            A value of either active or inactive returns stations currently set as active or inactive in the archive.
            Omitting this param returns all stations. e.g. status='active'
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of observations around a specific time.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['token'] = self.token

        return self._get_response('stations/nearesttime', kwargs)

    def latest(self, **kwargs):
        r""" Returns a dictionary of latest observations at a user specified location within a specified window. Users
        must specify at least one geographic search parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox',
        'cwa', 'nwsfirezone', 'gacc', or 'subgacc') to obtain observation data. Other parameters may also be included.
        See below for optional parameters. Also see the metadata() function for station IDs.

        Arguments:
        ----------
        within: string, required
            Represents the number of minutes which would return the latest ob within that time period. e.g. within='30'
            returns the first observation found within the last 30 minutes.
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: string, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        status: string, optional
            A value of either active or inactive returns stations currently set as active or inactive in the archive.
            Omitting this param returns all stations. e.g. status='active'
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of the latest time observations.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['token'] = self.token

        return self._get_response('stations/latest', kwargs)

    def precip(self, start, end, **kwargs):
        r""" Returns precipitation observations at a user specified location for a specified time. Users must specify at
        least one geographic search parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa',
        'nwsfirezone', 'gacc', or 'subgacc') to obtain observation data. Other parameters may also be included. See
        below mandatory and optional parameters. Also see the metadata() function for station IDs.

        Arguments:
        ----------
        start: string, mandatory
            Start date in form of YYYYMMDDhhmm. MUST BE USED WITH THE END PARAMETER. Default time is UTC
            e.g., start='201306011800'
        end: string, mandatory
            End date in form of YYYYMMDDhhmm. MUST BE USED WITH THE START PARAMETER. Default time is UTC
            e.g., end='201306011800'
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: list, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: list, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        status: string, optional
            A value of either active or inactive returns stations currently set as active or inactive in the archive.
            Omitting this param returns all stations. e.g. status='active'
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of precipitation observations.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['start'] = start
        kwargs['end'] = end
        kwargs['token'] = self.token

        return self._get_response('stations/precipitation', kwargs)

    def timeseries(self, start, end, **kwargs):
        r""" Returns a time series of observations at a user specified location for a specified time. Users must specify
        at least one geographic search parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa',
        'nwsfirezone', 'gacc', or 'subgacc') to obtain observation data. Other parameters may also be included. See
        below mandatory and optional parameters. Also see the metadata() function for station IDs.

        Arguments:
        ----------
        start: string, mandatory
            Start date in form of YYYYMMDDhhmm. MUST BE USED WITH THE END PARAMETER. Default time is UTC
            e.g., start='201306011800'
        end: string, mandatory
            End date in form of YYYYMMDDhhmm. MUST BE USED WITH THE START PARAMETER. Default time is UTC
            e.g., end='201306011800'
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: string, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        status: string, optional
            A value of either active or inactive returns stations currently set as active or inactive in the archive.
            Omitting this param returns all stations. e.g. status='active'
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of time series observations through the get_response() function.

        Raises:
        -------
            None.
        """

        self._check_geo_param(kwargs)
        kwargs['start'] = start
        kwargs['end'] = end
        kwargs['token'] = self.token

        return self._get_response('stations/timeseries', kwargs)

    def climatology(self, startclim, endclim, **kwargs):
        r""" Returns a climatology of observations at a user specified location for a specified time. Users must specify
        at least one geographic search parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa',
        'nwsfirezone', 'gacc', or 'subgacc') to obtain observation data. Other parameters may also be included. See
        below mandatory and optional parameters. Also see the metadata() function for station IDs.

        Arguments:
        ----------
        startclim: string, mandatory
            Start date in form of MMDDhhmm. MUST BE USED WITH THE ENDCLIM PARAMETER. Default time is UTC
            e.g. startclim='06011800' Do not specify a year
        endclim: string, mandatory
            End date in form of MMDDhhmm. MUST BE USED WITH THE STARTCLIM PARAMETER. Default time is UTC
            e.g. endclim='06011800' Do not specify a year
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: string, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        status: string, optional
            A value of either active or inactive returns stations currently set as active or inactive in the archive.
            Omitting this param returns all stations. e.g. status='active'
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of climatology observations through the get_response() function.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['startclim'] = startclim
        kwargs['endclim'] = endclim
        kwargs['token'] = self.token

        return self._get_response('stations/climatology', kwargs)

    def variables(self):
        """ Returns a dictionary of a list of variables that could be obtained from the 'vars' param in other functions.
        Some stations may not record all variables listed. Use the metadata() function to return metadata on each
        station.

        Arguments:
        ----------
            None.

        Returns:
        --------
            Dictionary of variables.

        Raises:
        -------
            None.

        """

        return self._get_response('variables', {'token': self.token})

    def climate_stats(self, startclim, endclim, type, **kwargs):
        r""" Returns a dictionary of aggregated yearly climate statistics (count, standard deviation,
        average, median, maximum, minimum, min time, and max time depending on user specified type) of a time series
        for a specified range of time at user specified location.  Users must specify at least one geographic search
        parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone', 'gacc', or 'subgacc')
        to obtain observation data. Other parameters may also be included. See below mandatory and optional parameters.
        Also see the metadata() function for station IDs.

        Arguments:
        ----------
        type: string, mandatory
            Describes what statistical values will be returned. Can be one of the following values:
            "avg"/"average"/"mean", "max"/"maximum", "min"/"minimum", "stdev"/"standarddeviation"/"std", "median"/"med",
            "count", or "all". "All" will return all of the statistics.
        startclim: string, mandatory
            Start date in form of MMDDhhmm. MUST BE USED WITH THE ENDCLIM PARAMETER. Default time is UTC
            e.g. startclim=06011800 Do not specify a year.
        endclim: string, mandatory
            End date in form of MMDDhhmm. MUST BE USED WITH THE STARTCLIM PARAMETER. Default time is UTC
            e.g. endclim=06011800 Do not specify a year.
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'.
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: string, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of aggregated climatology statistics.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['type'] = type
        kwargs['startclim'] = startclim
        kwargs['endclim'] = endclim
        kwargs['token'] = self.token

        return self._get_response('stations/climatology', kwargs)

    def time_stats(self, start, end, type, **kwargs):
        r""" Returns a dictionary of discrete time statistics (count, standard deviation, average, median, maximum,
        minimum, min time, and max time depending on user specified type) of a time series for a specified range of time
        at user specified location. Users must specify at least one geographic search parameter ('stid', 'state',
        'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone', 'gacc', or 'subgacc') to obtain observation data.
        Other parameters may also be included. See below mandatory and optional parameters. Also see the metadata()
        function for station IDs.

        Arguments:
        ----------
        type: string, mandatory
            Describes what statistical values will be returned. Can be one of the following values:
            "avg"/"average"/"mean", "max"/"maximum", "min"/"minimum", "stdev"/"standarddeviation"/"std", "median"/"med",
            "count", or "all". "All" will return all of the statistics.
        start: string, optional
            Start date in form of YYYYMMDDhhmm. MUST BE USED WITH THE END PARAMETER. Default time is UTC
            e.g. start=201506011800.
        end: string, optional
            End date in form of YYYYMMDDhhmm. MUST BE USED WITH THE START PARAMETER. Default time is UTC
            e.g. end=201506011800.
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        showemptystations: string, optional
            Set to '1' to show stations even if no obs exist that match the time period. Stations without obs are
            omitted by default.
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: list, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of discrete time statistics.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['type'] = type
        kwargs['start'] = start
        kwargs['end'] = end
        kwargs['token'] = self.token

        return self._get_response('stations/statistics', kwargs)

    def metadata(self, **kwargs):
        r""" Returns the metadata for a station or stations. Users must specify at least one geographic search parameter
        ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone', 'gacc', or 'subgacc') to obtain
        observation data. Other parameters may also be included. See below for optional parameters.

        Arguments:
        ----------
        complete: string, optional
            A value of 1 or 0. When set to 1, an extended list of metadata attributes for each returned station is
            provided. This result is useful for exploring the zones and regions in which a station resides.
            e.g. complete='1'
        sensorvars: string, optional
            A value of 1 or 0. When set to 1, a complete history of sensor variables and period of record is given for
            each station. e.g. sensorvars='1'
        obrange: string, optional
            Filters metadata for stations which were in operation for a specified time period. Users can specify one
            date or a date range. Dates are in the format of YYYYmmdd. e.g. obrange='20150101',
            obrange='20040101,20060101'
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: string, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables function to see a list of sensor vars.
        status: string, optional
            A value of either active or inactive returns stations currently set as active or inactive in the archive.
            Omitting this param returns all stations. e.g. status='active'
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            A dictionary of metadata.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['token'] = self.token

        return self._get_response('stations/metadata', kwargs)

    def latency(self, start, end, **kwargs):
        r""" Returns data latency values for a station based on a start and end date/time. Users must specify at least
        one geographic search parameter ('stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone',
        'gacc', or 'subgacc') to obtain observation data. Other parameters may also be included. See below mandatory and
        optional parameters. Also see the metadata() function for station IDs.

        Arguments:
        ----------
        start: string, mandatory
            Start date in form of YYYYMMDDhhmm. MUST BE USED WITH THE END PARAMETER. Default time is UTC
            e.g. start='201506011800'
        end: string, mandatory
            End date in form of YYYYMMDDhhmm. MUST BE USED WITH THE START PARAMETER. Default time is UTC
            e.g. end='201506011800'
        stats: string, optional
            Describes what statistical values will be returned. Can be one of the following values:
            "avg"/"average"/"mean", "max"/"maximum", "min"/"minimum", "stdev"/"standarddeviation"/"std", "median"/"med",
            "count", or "all". "All" will return all of the statistics. e.g. stats='avg'
        obtimezone: string, optional
            Set to either UTC or local. Sets timezone of obs. Default is UTC. e.g. obtimezone='local'
        stid: string, optional
            Single or comma separated list of MesoWest station IDs. e.g. stid='kden,kslc,wbb'
        county: string, optional
            County/parish/borough (US/Canada only), full name e.g. county='Larimer'
        state: string, optional
            US state, 2-letter ID e.g. state='CO'
        country: string, optional
            Single or comma separated list of abbreviated 2 or 3 character countries e.g. country='us,ca,mx'
        radius: list, optional
            Distance from a lat/lon pt or stid as [lat,lon,radius (mi)] or [stid, radius (mi)]. e.g. radius="-120,40,20"
        bbox: string, optional
            Stations within a [lon/lat] box in the order [lonmin,latmin,lonmax,latmax] e.g. bbox="-120,40,-119,41"
        cwa: string, optional
            NWS county warning area. See http://www.nws.noaa.gov/organization.php for CWA list. e.g. cwa='LOX'
        nwsfirezone: string, optional
            NWS fire zones. See http://www.nws.noaa.gov/geodata/catalog/wsom/html/firezone.htm for a shapefile
            containing the full list of zones. e.g. nwsfirezone='LOX241'
        gacc: string, optional
            Name of Geographic Area Coordination Center e.g. gacc='EBCC' See http://gacc.nifc.gov/ for a list of GACCs.
        subgacc: string, optional
            Name of Sub GACC e.g. subgacc='EB07'
        vars: string, optional
            Single or comma separated list of sensor variables. Will return all stations that match one of provided
            variables. Useful for filtering all stations that sense only certain vars. Do not request vars twice in
            the query. e.g. vars='wind_speed,pressure' Use the variables() function to see a list of sensor vars.
        units: string, optional
            String or set of strings and pipes separated by commas. Default is metric units. Set units='ENGLISH' for
            FREEDOM UNITS ;) Valid  other combinations are as follows: temp|C, temp|F, temp|K; speed|mps, speed|mph,
            speed|kph, speed|kts; pres|pa, pres|mb; height|m, height|ft; precip|mm, precip|cm, precip|in; alti|pa,
            alti|inhg. e.g. units='temp|F,speed|kph,metric'
        groupby: string, optional
            Results can be grouped by key words: state, county, country, cwa, nwszone, mwsfirezone, gacc, subgacc
            e.g. groupby='state'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of latency data.

        Raises:
        -------
            None.

        """

        self._check_geo_param(kwargs)
        kwargs['start'] = start
        kwargs['end'] = end
        kwargs['token'] = self.token

        return self._get_response('stations/latency', kwargs)

    def networks(self, **kwargs):
        r""" Returns the metadata associated with the MesoWest network ID(s) entered. Leaving this function blank will
        return all networks in MesoWest.

        Arguments:
        ----------
        id: string, optional
            A single or comma-separated list of MesoNet network categories. e.g. ids='1,2,3'
        shortname: string, optional
            A single or comma-separated list of abbreviations or short names. e.g shortname='DUGWAY,RAWS'
        sortby: string, optional
            Determines how to sort the returned networks. The only valid value is 'alphabet' which orders the results
            in alphabetical order. By default, networks are sorted by ID. e.g. sortby='alphabet'
        timeformat: string, optional
            A python format string for returning customized date-time groups for observation times. Can include
            characters. e.g. timeformat='%m/%d/%Y at %H:%M'

        Returns:
        --------
            Dictionary of network descriptions.

        Raises:
        -------
            None.

        """

        kwargs['token'] = self.token

        return self._get_response('networks', kwargs)

    def networktypes(self, **kwargs):
        r""" Returns the network type metadata depending on the ID specified. Can be left blank to return all network
        types.

        Arguments:
        ----------
        id: string, optional
            A single or comma-separated list of MesoNet categories. e.g.: type_ids='1,2,3'

        Returns:
        --------
            Dictionary of network type descriptions.

        Raises:
        -------
            None.

        """

        kwargs['token'] = self.token

        return self._get_response('networktypes', kwargs)

        # Leaving off qctypes until I get the qctypes response to give response code/msg

        # def qc_types(self, **kwargs):
        #     r""" Returns the quality control and internal consistency test types used by MesoWest. These include MesoWest
        #     checks and MADIS checks. Leaving this blank will return all QC types.
        #
        #     Arguments:
        #     ----------
        #     id: string, optional
        #         A single or comma-separated list of test ids. e.g. id='1,2,3'
        #     shortname: string, optional
        #         A single or comma-separated list of MesoWest QC/IC test shortnames. e.g. shortname='mw_range_check'
        #
        #
        #     Returns:
        #     --------
        #         Dictionary of QC types.
        #
        #     Raises:
        #     -------
        #         None.
        #
        #     """
        #
        #     kwargs['token'] = self.token
        #
        #     return self._get_response('qctypes', kwargs)
