# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:06:06 2016

@author: saf537
"""
import pandas as pd
import matplotlib.pyplot as plt
import urllib2
import requests
import json
import xml.etree.ElementTree as ET
import random

master = pd.read_csv('nUrve_master_0321.csv')
sampleB = pd.read_csv('pilotB_sample1.csv')


group_keys = ['b901157a817d3058367b22fdabcc6596'] # Sara


master['uniqueLatLon'] = master.apply(lambda x: str(x.GPS_LAT)+'|'+str(x.GPS_LON), axis=1)

"""
This is how we created the file with unique locations to match with the parcels data.


test = pd.DataFrame(master.groupby(['uniqueLatLon', 'GPS_LAT','GPS_LON'])['ID'].count())
test.reset_index(inplace=True)
test.to_csv('UniqueLocations.csv')

"""

nUrv_add = pd.read_csv('nUrveLocAdd.csv')

"""
1119 8th Ave S, Seattle, WA

"""

Stnum = '1119' #ST_NUM
Stname = '8th'
st_nm_su = 'Ave' #ST_NAME_SU
LAT = '47.6085'
LON = '-122.3295' # sample -122.3295
City = 'Seattle'
State = 'WA'
Zip = '9810' # ZIPCODE
#str_call = 'http://api.walkscore.com/score?format=xml&address=1119%8th%20Avenue%20Seattle%20WA%2098101&lat='+LAT+'&lon='+LON+'&wsapikey='+mykey+'&format=JSON'

nUrv_add = nUrv_add.rename(index=str, columns={"uniqueLatL": "uniqueLatLon"});
data = pd.merge(nUrv_add,master,on='uniqueLatLon')

for key in group_keys:
    rows = random.sample(data.index, 5000)
    samp = data.ix[rows]
    data = data.drop[rows]
    for i in range(0,1):
        Stnum = samp['ST_NUM'].ix[rows[i]] #ST_NUM
        Stname = samp['ST_NAME'].ix[rows[i]] 
        st_nm_su = samp['ST_NAME_SU'].ix[rows[i]]
        LAT = samp['Latitude'].ix[rows[i]]
        LON = samp['Longitude'].ix[rows[i]] # sample -122.3295
        City = 'Boston'
        State = 'MA'
        Zip = samp['ZIPCODE'].ix[rows[i]] # ZIPCODE
        
        call = 'http://api.walkscore.com/score?format=xml&address='+str(Stnum)+'%'+str(Stname)+'%20'+str(st_nm_su)+'%20'+str(City)+'%20'+str(State)+'%20'+str(Zip)+'&lat='+str(LAT)+'&'+'lon='+str(LON)+'&wsapikey='+str(key)+'&format=XML' 
        #url_web = 'http://api.walkscore.com/score?format=xml&address=1119%8th%20Avenue%20Seattle%20WA%2098101&lat=47.6085&lon=-122.3295&wsapikey='+key+'&format=XML'
        request = requests.get(call)
        
        tree = ET.fromstring(request.content)
        for child in tree.findall('{http://walkscore.com/2008/results}walkscore'):
            print child.text #This is the walkscore

    

#url = 'https://nycopendata.socrata.com/views/%s' % (dataId)


#request = urllib2.urlopen(call)
#metadata = json.loads(request.read())
