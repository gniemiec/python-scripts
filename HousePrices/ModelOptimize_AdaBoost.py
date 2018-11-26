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
    'n_estimators': [1300],
    'learning_rate': [3.1,3.5,4,4.5,5],
    'loss': ['linear'],
}
'''
{'learning_rate': 3.25, 'loss': 'linear', 'n_estimators': 525}
{'learning_rate': 1, 'loss': 'linear', 'n_estimators': 1000}
'linear', 'square', 'exponential'
{'n_estimators': 1100, 'loss': 'linear', 'learning_rate': 2.5}
{'n_estimators': 1200, 'loss': 'linear', 'learning_rate': 3}
{'n_estimators': 1300, 'loss': 'linear', 'learning_rate': 3}
{'n_estimators': 1300, 'loss': 'linear', 'learning_rate': 3.1}
{'n_estimators': 1300, 'loss': 'linear', 'learning_rate': 3.1}
'''
# Create a based model
ab = AdaBoostRegressor(random_state = 42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = ab, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X, train_y)

download_dir = "bestAdaBoost.csv" #where you want the file to be downloaded to 

csv = open(download_dir, "w") 
#"w" indicates that you're writing strings to the file

columnTitleRow = str(grid_search.best_params_)
csv.write(columnTitleRow)

