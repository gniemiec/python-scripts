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
   'n_iter' : [2],
#Maximum number of iterations. Default is 300.
    'tol' : [0.001,0.01],
#Stop the algorithm if w has converged. Default is 1.e-3.
    'alpha_1' : [0.000001,0.00001],
#Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter. Default is 1.e-6
    'alpha_2' :[0.000001,0.00001],
#Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Default is 1.e-6.
    'lambda_1' : [0.000001,0.00001],
#Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter. Default is 1.e-6.
   'lambda_2' : [0.000001,0.00001],
#Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Default is 1.e-6
    'compute_score' : [False,True],
#If True, compute the objective function at each step of the model. Default is False
    'fit_intercept' : [True],
#whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered). Default is True.
    'normalize' : [False]
}
'''
'''
# Create a based model
estimator = linear_model.BayesianRidge()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_X, train_y)

download_dir = "bestBayesRidge.csv" #where you want the file to be downloaded to 

csv = open(download_dir, "w") 
#"w" indicates that you're writing strings to the file

columnTitleRow = str(grid_search.best_params_)
csv.write(columnTitleRow)

