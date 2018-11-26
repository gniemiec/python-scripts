import numpy as np 
import pandas as pd 
#%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from numpy import nan
from sklearn.preprocessing import scale
pd.set_option("max.columns",300)
import os

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge


train  = pd.read_csv('train.csv')

train_X = train.drop (['Id','SalePrice'], axis = 1)
train_y = train.SalePrice


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'n_estimators': [1000],
    'max_depth': [2,3],
    'min_child_weight': [1,2],
    'subsample' : [0.7,0.8],
    'colsample_bytree': [0.7,0.8],
    'reg_alpha' : [0.01,0.1,1,3],
    'learning_rate' : [0.01,0.1,1,3],
}
'''

'''
# Create a based model
estimator = XGBRegressor(random_state = 42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X, train_y)

download_dir = "bestXGBoost.csv" #where you want the file to be downloaded to 

csv = open(download_dir, "w") 
#"w" indicates that you're writing strings to the file

columnTitleRow = str(grid_search.best_params_)
csv.write(columnTitleRow)

