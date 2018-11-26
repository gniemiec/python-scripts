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

# Create the parameter grid based on the results of random search 
param_grid = {
    'n_estimators' : [950,1000,1050], 
    'learning_rate' : [0.05],
    'max_depth' : [3],
    'max_features' : ['sqrt'],
    'min_samples_leaf' : [2], 
    'min_samples_split' : [2], 
    'loss' : ['huber'],
    'alpha' : [0.3,0.4,0.6,0.8,0.9]
}
'''
param_grid = {
    'n_estimators' : [1000], 
    'learning_rate' : [0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2],
    'max_depth' : [2,3],
    'max_features' : ['sqrt'],
    'min_samples_leaf' : [1,2,3], 
    'min_samples_split' : [2,3], 
    'loss' : ['huber']
}

'''
#{'loss': 'huber', 'learning_rate': 0.05, 'min_samples_leaf': 2, 'n_estimators': 1000, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 3}
#{'loss': 'huber', 'learning_rate': 0.05, 'min_samples_leaf': 2, 'n_estimators': 1000, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 3}
#{'loss': 'huber', 'learning_rate': 0.05, 'min_samples_leaf': 2, 'n_estimators': 1000, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 3}
# Create a based model
estimator = GradientBoostingRegressor(random_state = 42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X, train_y)

download_dir = "bestGBoost.csv" #where you want the file to be downloaded to 

csv = open(download_dir, "w") 
#"w" indicates that you're writing strings to the file

columnTitleRow = str(grid_search.best_params_)
csv.write(columnTitleRow)

