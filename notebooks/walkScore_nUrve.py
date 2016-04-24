# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:06:06 2016

@author: saf537
"""
import pandas as pd
import requests
import glob
import xml.etree.ElementTree as ET
import random
import numpy as np
from xml.etree.ElementTree import ParseError

random.seed('2339')

master = pd.read_csv('nUrve_master_0321.csv')
sampleB = pd.read_csv('pilotB_sample1.csv')


group_keys = ['b901157a817d3058367b22fdabcc6596'] # Sara


master['uniqueLatLon'] = master.apply(lambda x: str(x.GPS_LAT)+'|'+str(x.GPS_LON), axis=1)

"""
This is how we created the file with unique locations to match the parcels data.


test = pd.DataFrame(master.groupby(['uniqueLatLon', 'GPS_LAT','GPS_LON'])['ID'].count())
test.reset_index(inplace=True)
test.to_csv('UniqueLocations.csv')

"""



"""
SAMPLE CALL

1119 8th Ave S, Seattle, WA

Stnum = '1119' #ST_NUM
Stname = '8th'
st_nm_su = 'Ave' #ST_NAME_SU
LAT = '47.6085'
LON = '-122.3295' # sample -122.3295
City = 'Seattle'
State = 'WA'
Zip = '9810' # ZIPCODE
#str_call = 'http://api.walkscore.com/score?format=xml&address=1119%8th%20Avenue%20Seattle%20WA%2098101&lat='+LAT+'&lon='+LON+'&wsapikey='+mykey+'&format=JSON'

"""
nUrv_add = pd.read_csv('nUrveLocAdd.csv')
nUrv_add = nUrv_add.rename(index=str, columns={"uniqueLatL": "uniqueLatLon"});
nUrv_add = nUrv_add[['uniqueLatLon',u'PID','ST_NUM', u'ST_NAME', u'ST_NAME_SU', u'UNIT_NUM',u'ZIPCODE', u'full_addre', u'Latitude', u'Longitude']]



# get data file names
path =r'/home/saf537/Documents/CUSP/Spring/MLC/ml-project-2016/notebooks/data_walk'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
#big_frame = big_frame[['uniqueLatLon',u'PID','ST_NUM', u'ST_NAME', u'ST_NAME_SU', u'UNIT_NUM',u'ZIPCODE', u'full_addre', u'Latitude', u'Longitude']]



"""

CAREFUL!!!

"""
ind = []
arr = np.array(big_frame['uniqueLatLon'])
for i in range(0,len(nUrv_add)):
    ind.append(np.array(nUrv_add['uniqueLatLon'].iloc[i]) not in arr)
    
nUrv_add = nUrv_add[ind]
nUrv_add.reset_index()

df = pd.DataFrame([],columns = nUrv_add.columns)
df['walkscore'] = []

spl_size = 5000


for key in group_keys:
    rows = random.sample(nUrv_add.index, spl_size)
    samp = nUrv_add.ix[rows]

    nUrv_add = nUrv_add.drop(rows)
    samp['walkscore'] = np.zeros(len(samp))
    wklist = []

    for i in range(0,spl_size):
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
        try:
            tree = ET.fromstring(request.content)
            for child in tree.findall('{http://walkscore.com/2008/results}walkscore'):
                # print "Am I coming here when there is an error?"
                wklist.append(int(child.text))
                # print child.text #This is the walkscore
                # print tree
        except ParseError as e:
            print "Error likely due to absence of walkscore element in response"
            wklist.append(np.NAN)
    
    samp['walkscore'] = np.zeros(len(wklist))
    samp['walkscore'] = wklist
    df.append(samp)

n_files = str(len(filenames)+1)

samp = samp[['uniqueLatLon','walkscore']]
#data = pd.merge(samp,master,on='uniqueLatLon')
#data = data[data['walkscore'].isnull()!=True]
samp = samp[samp['walkscore'].isnull()!=True]
samp.to_csv('data_walk/walkscores'+n_files+'.csv')
data['GPS_DATETIMESTAMP'] = pd.to_datetime(data['GPS_DATETIMESTAMP'])

"""
# Create linear regression object
regr = lm.LinearRegression()

# Train the model using the training sets
regr.fit(np.array(data['GPS_Speed']).reshape(len(data['GPS_Speed']),1), np.adarray(data['walkscore']).reshape(len(data['walkscore']),1))
print('Coefficients: \n', regr.coef_)
y_p = regr.predict(np.array(data['GPS_Speed']).reshape(len(data['GPS_Speed']),1))
y = np.array(data['walkscore']).reshape(len(data['walkscore']),1)
print("Residual sum of squares: %.2f" % np.mean((y_p - y) ** 2))

# Plot outputs
plt.scatter(data['GPS_Speed'], y,  color='black')
plt.plot(data['GPS_Speed'], y_p, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
"""

#url = 'https://nycopendata.socrata.com/views/%s' % (dataId)


#request = urllib2.urlopen(call)
#metadata = json.loads(request.read())
