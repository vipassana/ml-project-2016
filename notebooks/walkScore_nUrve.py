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

master = pd.read_csv('nUrve_master_0321.csv')
sampleB = pd.read_csv('pilotB_sample1.csv')

mykey = 'b901157a817d3058367b22fdabcc6596'


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

call = 'http://api.walkscore.com/score?format=xml&address='+Stnum+'%'+Stname+'%20'+st_nm_su+'%20'+City+'%20'+State+'%20'+Zip+'&lat='+LAT+'&'+'lon='+LON+'&wsapikey='+mykey+'&format=json'
url_web = 'http://api.walkscore.com/score?format=xml&address=1119%8th%20Avenue%20Seattle%20WA%2098101&lat=47.6085&lon=-122.3295&wsapikey='+mykey+'&format=XML'
request = requests.get(url_web)

tree = ET.fromstring(request.content)
for child in tree.findall('{http://walkscore.com/2008/results}walkscore'):
    print child.text #This is the walkscore





#url = 'https://nycopendata.socrata.com/views/%s' % (dataId)


#request = urllib2.urlopen(call)
#metadata = json.loads(request.read())
