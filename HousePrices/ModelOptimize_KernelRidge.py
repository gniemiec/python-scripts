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
    'alpha': [0,0.000001,0.00005,0.00001,0.001,0.01,0.05,0.07,0.1,0.2]
}
# Create a based model
estimator = KernelRidge()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X, train_y)

download_dir = "bestKernelRidge.csv" #where you want the file to be downloaded to 

csv = open(download_dir, "w") 
#"w" indicates that you're writing strings to the file

columnTitleRow = str(grid_search.best_params_)
csv.write(columnTitleRow)

