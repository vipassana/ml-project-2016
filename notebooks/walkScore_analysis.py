# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:39:08 2016

@author: MLC team of awesomness.
"""
from sklearn.cross_validation import train_test_split
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import pandas as pd
from pandas.stats.api import ols
import glob
import numpy as np
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn import linear_model


"""
Data reading and cleaning
"""

# get data file names
path =r'/home/saf537/Documents/CUSP/Spring/MLC/ml-project-2016/notebooks/data_walk'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame['GPS_DATETIMESTAMP'] = pd.to_datetime(big_frame['GPS_DATETIMESTAMP'])
big_frame['hour'] = big_frame['GPS_DATETIMESTAMP'].apply(lambda x: x.hour)

cols_to_norm = [ u'GPS_LAT', u'GPS_LON', u'GPS_Speed', u'GPS_Alt',u'GPS_Sats', u'GPS_Fix', u'GPS_Quality', u'AMB_Temp', u'AMB_Humd',u'AMB_Lux', u'AMB_Snd', u'AMB_SndMin', u'AMB_SndMax', u'AMB_SndMea',u'RDQ_AcX', u'RDQ_AcXMin', u'RDQ_AcXMax', u'RDQ_AcXMea', u'RDQ_AcY',u'RDQ_AcYMin', u'RDQ_AcYMax', u'RDQ_AcYMea',u'RDQ_AcZMax', u'RDQ_AcZMea']
big_frame[cols_to_norm] = big_frame[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

# Split data into training and testing
train_size = int(0.6*len(big_frame))
rows = random.sample(big_frame.index, train_size)
samp_train = big_frame.ix[rows]
y_train = samp_train['walkscore']

var_cols = ['hour',u'GPS_Speed',u'AMB_Temp', u'AMB_SndMea', u'RDQ_AcZMea']#u'GPS_Speed', u'GPS_Alt',u'AMB_Temp', u'AMB_Humd', u'AMB_SndMea',u'RDQ_AcXMea', u'RDQ_AcY', u'RDQ_AcYMea', u'RDQ_AcZMea']
X_train, X_test, y_train, y_test = train_test_split(big_frame[var_cols], big_frame['walkscore'], test_size=0.33, random_state=42)

"""
Feature selection
"""

X, y = X_train, y_train
print(X.shape)
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
print(clf.feature_importances_  )
#model = SelectFromModel(clf, prefit=True)
#X_new = model.transform(X)
#X_new.shape               


pca = PCA(n_components=5)
X = np.array(X)
pca.fit(X)
print("variance explained via the first and second components:\n" , pca.explained_variance_)
print("principal components:\n", pca.components_)

plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal');

"""
Linear model
"""

result=ols(y=y_train,x=pd.DataFrame(X_train))
R_2_IS=result.r2  # get R2
OLS_coef=result.beta

#Out of sample 
a=np.array(X_test[var_cols])  #makes sure conver pd data to np array
b=np.array(result.beta) #makes sure conver pd data to np array
print('OLS regression coefficients={0}'.format(b))
c=np.sum(a*b[0:-1],axis=1)+b[-1] #b is estimated coefficients, a is prediction data, b[-1] is intercept. This is for predicted y
error=y_test-c # y_predict is real value, c is the value we guessed
R_2_OS=1-error.var()/y_test.var() # this is out of sample R2
print("The R-squared we found for in-sample (IS) OLS is: {0}".format(R_2_IS))
print("The R-squared we found for out-of-sample (OS) OLS is: {0}".format(R_2_OS))

print(result.summary)

result=ols(y=y,x=pd.DataFrame(X))

"""
Bayessian regression: Lasso
"""

Lasso=linear_model.Lasso(fit_intercept=True,alpha=1) #try Ridge with an arbitrary regularization parameter lambda=1

Lasso.fit(X_train,y_train)
# In the sample:
p_IS=Lasso.predict(X_train)
err_IS=p_IS-y_train
R_2_IS_Lasso=1-np.var(err_IS)/np.var(y_train)
print("The R-squared we found for IS Lasso is: {0}".format(R_2_IS_Lasso))

Lasso_coef=Lasso.coef_
############################################################################    
    
#Out of sample
p_OS=Lasso.predict(X_test)
err_OS=p_OS-y_test
R_2_OS_Ridge=1-np.var(err_OS)/np.var(y_test)
print("The R-squared we found for OS Ridge is: {0}".format(R_2_OS_Ridge))


"""
Neural networks
"""


