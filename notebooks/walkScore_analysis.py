# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:39:08 2016

@author: MLC team of awesomness.
"""
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from pandas.stats.api import ols
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn import linear_model


"""
Data reading and cleaning
"""

def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

# get data file names
data = pd.read_csv('df_walkscore.pkl')
data['GPS_DATETIMESTAMP'] = pd.to_datetime(data['GPS_DATETIMESTAMP'])
data['hour_slot'] = data['GPS_DATETIMESTAMP'].apply(lambda x: x.hour)
data = data.groupby(['road_id','walkscore','uniqueLatLon']).aggregate('mean')
data = data.reset_index()
norm_road = ['is_dark','is_loud','bumpflag']
for col in norm_road:
    data[col] = data[col]/data['road_length']
data["hour_slot"] = binning(col = data["hour_slot"], cut_points = [11.,13.,15.], labels = ["morning","noon","afternoon","later_afternoon"])

data['morning'] = np.nan
data['noon'] = np.nan
data['afternoon'] = np.nan
data['later_afternoon'] = np.nan
data[["morning","noon","afternoon","later_afternoon"]] = pd.get_dummies(data["hour_slot"])
cols_to_norm1 = [ 'walkscore',u'AMB_Lux',  u'AMB_SndMea','acel','is_dark','is_loud','bumpflag']
data[cols_to_norm1] = (data[cols_to_norm1])/(data[cols_to_norm1].max())



# Split data into training and testing
var_cols = ['morning','noon','afternoon','later_afternoon',u'AMB_Lux', u'AMB_SndMea',u'acel', u'is_dark', u'is_loud', u'bumpflag',]#u'GPS_Speed', u'GPS_Alt',u'AMB_Temp', u'AMB_Humd', u'AMB_SndMea',u'RDQ_AcXMea', u'RDQ_AcY', u'RDQ_AcYMea', u'RDQ_AcZMea']
X_train, X_test, y_train, y_test = train_test_split(data[var_cols], data['walkscore'], test_size=0.33, random_state=42)

"""
Feature selection
"""



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

result=ols(y=y_test,x=pd.DataFrame(X_test))
print(result.summary)


"""
Bayessian regression: Lasso
"""

Lasso=linear_model.Lasso(fit_intercept=True,alpha=0.001) #try Ridge with an arbitrary regularization parameter lambda=1

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
R_2_OS_Lasso=1-np.var(err_OS)/np.var(y_test)
print("The R-squared we found for OS Lasso is: {0}".format(R_2_OS_Lasso))


"""
Bayessian regression: Ridge
"""

Ridge=linear_model.Ridge(fit_intercept=True) #try Ridge with an arbitrary regularization parameter lambda=1

Ridge.fit(X_train,y_train)
# In the sample:
p_IS=Ridge.predict(X_train)
err_IS=p_IS-y_train
R_2_IS_Ridge=1-np.var(err_IS)/np.var(y_train)
print("The R-squared we found for IS Ridge is: {0}".format(R_2_IS_Ridge))

Ridge_coef=Ridge.coef_
############################################################################    
    
#Out of sample
p_OS=Ridge.predict(X_test)
err_OS=p_OS-y_test
R_2_OS_Ridge=1-np.var(err_OS)/np.var(y_test)
print("The R-squared we found for OS Ridge is: {0}".format(R_2_OS_Ridge))



"""
Neural networks
"""
from function_approximator import FunctionApproximator
fa = FunctionApproximator(n_out=1, n_hidden=5,n_in=10)
x_nn = np.array(X_train).reshape((len(X_train),10))
y_nn = np.array(y_train).reshape((len(y_train),))
fa.train(x_nn,y_nn,learning_rate=0.05, n_epochs=2000000, report_frequency=500000)
Y_pred = fa.get_y_pred()
is_err_nn = (Y_pred-np.array(y_nn).reshape(len(y_nn),1))
R_2_IS_nn = 1-np.var(is_err_nn)/np.var(np.array(y_nn))
print(R_2_IS_nn)

y_os_nn = fa.predict_model(np.array(X_test).reshape((len(X_test),10)))
os_err_nn = (y_os_nn-np.array(y_test).reshape((len(y_test),1)))
R_2_OS_nn = 1-np.var(os_err_nn)/np.var(np.array(y_nn))
print(R_2_OS_nn)




#fig = plt.figure(figsize=[12,8])
#plt.plot(X, Y, 'o');
#plt.plot(X, Y_pred, 'x');
#[w,b]=fa.get_weights()
#print("w:", w)

