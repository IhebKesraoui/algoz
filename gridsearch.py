#!/usr/bin/env python3

import logging
import os
from pathlib import Path
import shutil
from datetime import datetime

import pandas as pd
from sklearn.model_selection import GridSearchCV

import nanoz as nz

# TODO: don't use this module, will be integrated for Algoz 3.0.0

__author__ = "Matthieu Nugue"
__license__ = "Apache License 2.0"
__version__ = "1.2.1"  # TODO: version() from utils
__maintainer__ = "Matthieu Nugue"
__email__ = "matthieu.nugue@nanoz-group.com"
__status__ = "Development"


pd.options.mode.chained_assignment = None  # Disable warning SettingWithCopyWarning, default='warn'

# Full execution time
start_time = datetime.now()


#########
# Input #
#########

# Path to the root directory of the project
path_to_Algoz = nz.io.get_root_path('Algoz')

# Load code parameters
cp, train_parameters_path = nz.io.load_config(path_to_Algoz, 'grid_search_parameters.json')


##########
# Output #
##########

# Results directories
dir_to_save = Path(cp.output_path, start_time.strftime("%Y%m%d%H%M%S")+'_'+cp.model_name)
os.makedirs(dir_to_save)
shutil.copyfile(train_parameters_path, Path(dir_to_save, os.path.basename(train_parameters_path)))

# File loggers
flogger = nz.loggers.file_logger(Path(dir_to_save, 'grid_search.log'))
logging.debug('Execution started at {0}'.format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
logging.info('Results directory created at: {0}'.format(dir_to_save))


#################
# Preprocessing #
#################

# Data_preparation
df_csv, df_x, df_y = nz.preprocessing._data_preparation(cp)

# Features extraction
df_x, df_y = nz.preprocessing._features_extraction(df_x, df_y, cp)

# Split data into train and test subsets
df_x_train, df_x_test, df_y_train, df_y_test = nz.preprocessing._splitting(df_x, df_y, cp)


###################################
# Optimization of hyperparameters #
###################################

start_time_tr = datetime.now()

# Load model hyperparameters
hyperparams = nz.io.load_hyperparameters(path_to_Algoz, cp.model_name+'.json')

# Optimization of hyperparameters
scoring = nz.performance_evaluation.create_scorer(cp.metrics_name)
model = nz.modeling.get_regression_model(cp.model_name, hyperparameters=hyperparams)
grid_search = GridSearchCV(model, param_grid=cp.param_gridsearch, scoring=scoring, refit=False, verbose=1)
grid_search.fit(df_x_train, df_y_train.values.ravel())

# Report grid search results
df_grid_search = nz.performance_evaluation.report_grid_search(grid_search, cp.metrics_name)

# Save hyperparameters and results of the grid search
nz.io.save_file(hyperparams, Path(dir_to_save, 'static_hyperparameters.json'))
nz.io.save_file(cp.param_gridsearch, Path(dir_to_save, 'grid_search_hyperparameters.json'))
nz.io.save_file(grid_search, Path(dir_to_save, 'grid_search.pkl'))
nz.io.save_file(df_grid_search, Path(dir_to_save, 'grid_search.csv'))

# Execution time
nz.utils.execution_time(start_time_tr, name='optimization of hyperparameters')

# Full execution time
nz.utils.execution_time(start_time, name='Full')
