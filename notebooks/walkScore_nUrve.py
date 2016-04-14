# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:06:06 2016

@author: saf537
"""
import pandas as pd
import matplotlib.pyplot as plt

master = pd.read_csv('nUrve_master_0321.csv')  
sampleB = pd.read_csv('pilotB_sample1.csv')
sampleB['lat_mod'] = sampleB['GPS_LAT']

mykey = 'b901157a817d3058367b22fdabcc6596'
#LAT = 'bla'
#LON = 'bla' # sample -122.3295
#str_call = 'http://api.walkscore.com/score?format=xml&address=1119%8th%20Avenue%20Seattle%20WA%2098101&lat='+LAT+'&lon='+LON+'&wsapikey='+mykey+'&format=JSON'

call = 'http://api.walkscore.com/score?format=xml&address=1119%8th%20Avenue%20Seattle%20WA%2098101&lat=47.6085&lon=-122.3295&wsapikey='+mykey

master['uniqueLatLon'] = master.apply(lambda x: str(x.GPS_LAT)+'|'+str(x.GPS_LON), axis=1)

test = pd.DataFrame(master.groupby(['uniqueLatLon', 'GPS_LAT','GPS_LON'])['ID'].count())
test.reset_index(inplace=True)
test.to_csv('UniqueLocations.csv')