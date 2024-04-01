import pandas as pd
import numpy as np
import xarray as xr

import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import random
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("c:\\Users\\nerea\\OneDrive\\Documentos\\EPFL_MASTER\\PDM\\Project\\PyalData")
from pyaldata import *



def clean_0d_array_fields_NC(df):
    """
    Loading v7.3 MAT files, sometimes scalers are stored as 0-dimensional arrays for some reason.
    This converts those back to scalars.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    for c in df.columns:
        if isinstance(df[c].values.all, np.ndarray) and all([arr.ndim == 0 for arr in df[c]]):
            df[c] = [arr.item() for arr in df[c]]

    return df


def mat2dataframe_NC(fname, shift_idx_fields=True, td_name = 'td_name'):
    try:
        mat = scipy.io.loadmat(fname, simplify_cells=True)
    except NotImplementedError:
        try:
            import mat73
        except ImportError:
            raise ImportError("Must have mat73 installed to load mat73 files.")
        else:
            mat = mat73.loadmat(fname)
    real_keys = [k for k in mat.keys() if not (k.startswith("__") and k.endswith("__"))]

    if td_name is None:
        if len(real_keys) == 0:
            raise ValueError("Could not find dataset name. Please specify td_name.")
        elif len(real_keys) > 1:
            raise ValueError("More than one datasets found. Please specify td_name.")

        assert len(real_keys) == 1

        td_name = real_keys[0]

    df = pd.DataFrame(mat[td_name])

    # Apply the function to all columns of the DataFrame
    df = df.map(replace_empty_list)
    df= clean_0d_array_fields_NC(df) #changed from the function in pyaldata to avoid errors when some rows were arrays and other were not.
    df = data_cleaning.clean_integer_fields(df)
    
    if shift_idx_fields:
        df = data_cleaning.backshift_idx_fields(df)
    
    return df 


#Auxiliar function to replace cells with [] by "None"
def replace_empty_list(value):
    return 0 if (
        (isinstance(value, list) and len(value) == 0) or
        (isinstance(value, np.ndarray) and value.size == 0)
    ) else value


def add_bad_idx(end_list, bad_list):
    
    """ The original data has saved the end index for the bad trials in a separate variable, 
    called "idx_bad". This function adds the bad trial end to the list of end indices, and sorts them
    
    Inputs:
        - end_list (array with most end indices)
        - bad_list (array or int corresponding to the bad reaches)

    Output: 
        - Array with all end indices combined and sorted.
    """


    if (isinstance(bad_list, int) or isinstance(bad_list, np.integer)) and bad_list == -1:

        pass

    elif (isinstance(bad_list, int) or isinstance(bad_list, np.integer)) and bad_list>-1:
        
        if (isinstance(end_list, int) or isinstance(end_list, np.integer)) and end_list>-1:
    
                end_list_ = []
                end_list_.append(end_list)
                end_list_.append(bad_list)
                end_list_.sort()
                end_list = end_list_

        else:
   
            end_list = list(end_list)
            end_list.append(bad_list)
            end_list.sort()

    else:
        
            if (isinstance(end_list, int) or isinstance(end_list, np.integer))and end_list>-1:
         
                end_list_ = []
                end_list_.append(end_list)
                end_list_.extend(bad_list)
                end_list_.sort()
                end_list = end_list_

            else:
            
                end_list = list(end_list)
                end_list.extend(bad_list)
                end_list.sort()

    return end_list




def find_bad_indices(row):

    """ The original data has saved the end index for the bad trials in a separate variable, 
        called "idx_bad". This function keeps a track of the bad trials. To be used inside Pandas apply function.
        
        Inputs:
            row of the data frame

        Output: 
            list of 1 and 0 corresponding to all the reaches of a trial.
        """

    idx_end_complete = row['idx_end_complete']
    idx_bad = row['idx_bad']
    
    bad_indices = []
    for trial in idx_end_complete:
        if isinstance(trial, list):
            trial_indices = [1 if idx in idx_bad else 0 for idx in trial]
            bad_indices.append(trial_indices)
        else:  # If it's not a list, it's a single trial
            # Check if idx_bad is iterable
            if hasattr(idx_bad, '__iter__'):
                bad_indices.append(1 if trial in idx_bad else 0)
            elif (isinstance(idx_bad,int) and trial == idx_bad):
                bad_indices.append(1)
            else:
                bad_indices.append(0)  # Assuming no bad trials if idx_bad is not iterable
    
    return bad_indices



def slice_between_points_NC(start_point, end_point, before, after):
    """
    Return a slice that starts before start_point_name and ends after end_point_name

    Parameters
    ----------
    trial : pd.Series
        a row from a trial_data dataframe
        representing a trial
    start_point_name : str
        name of the time point around which the interval starts
    end_point_name : str
        name of the time point around which the interval ends
    before : int
        number of time points before time point
    after : int
        number of time points after time point

    Returns
    -------
    slice object
    """
    start = start_point - before
    end = end_point + after + 1

    if np.isfinite(start):
        start = int(start)
    if np.isfinite(end):
        end = int(end)

    return slice(start, end)



def split_time_fields(df, start_margin = 5, end_margin = 10,ref_field = None):
    """ We want to split data for each reach inside trials.
    Only applies to the time varying fields.
    
    Intput:
        - df: DataFrame
        - start and end margin: (int) samples to include before and after the reach and end indices.
            Default 10.
        - ref_field: argument of function "get_time_varying_fields", can stay "None"
    
    Returns:
        - win_df: df with only the time varying fields split by the idx_reach and idx_end_complete"""

    time_fields = utils.get_time_varying_fields(df, ref_field)
    win_df = pd.DataFrame(index=range(len(df)), columns=time_fields)  # Creating DataFrame with columns based on time_fields

    for it, row in df.iterrows():
        idx_reach_values = row.idx_reach
        idx_end_values = row.idx_end_complete

        for t in time_fields:
            col = []
            t_row_values = row[t]
            for i, reach in enumerate(idx_reach_values):
                start = int(reach - start_margin)
                #end = int(idx_end_values[i] + end_margin)
                end = start + 75
                data = t_row_values[start:end]
                col.append(data)
            
            win_df.loc[it, t] = [col]
            
    return win_df





def build_tidy_df(df, start_margin = 10, end_margin = 10,ref_field = None):

    """ We want to split data for each reach inside trials.
    Only applies to the time varying fields.
    
    Intput:
        - df: DataFrame
        - start and end margin: (int) samples to include before and after the reach and end indices.
            Default 10.
        - ref_field: argument of function "get_time_varying_fields", can stay "None"
    
    Internal variable:
        - win_df: df with only the time varying fields split by the idx_reach and idx_end_complete
        
    From the dataframe split by reach number and filtered, we 
    now prepare the dataframe to be fed to the models.
    
    Return:
        df0: DataFrame with tidy data
    
    """

    time_fields = utils.get_time_varying_fields(df, ref_field)
    win_df = pd.DataFrame(index=range(len(df)), columns=time_fields)  # Creating DataFrame with columns based on time_fields

    for it, row in df.iterrows():
        idx_reach_values = row.idx_reach
        idx_end_values = row.idx_end_complete

        for t in time_fields:
            col = []
            t_row_values = row[t]
            for i, reach in enumerate(idx_reach_values):
                start = int(reach - start_margin)
                #end = int(idx_end_values[i] + end_margin)
                end = start + 75 #changed on 27/03 to have all trials with same length, easier shape for the models.
                data = t_row_values[start:end]
                col.append(data)
            
            win_df.loc[it, t] = [col]

    cols_to_search = ['index', 'num', 'type', 'KUKAPos']
    cols_to_keep = [c for c in cols_to_search if c in df.columns]

    for col in cols_to_keep:
        win_df[col] = df[col]


    rows_0 = []
    for trial_num, trial in enumerate(win_df[time_fields[0]]):
        for reach_num, reach in enumerate(trial[0]):
            for timestamp, values in enumerate(reach):
                row = {
                     'num': win_df['num'][trial_num],
                    'type': win_df['type'][trial_num],
                    #'KUKAPos': win_df['KUKAPos'][trial_num][reach_num], --> check if I need it, only in some data
                    'trial_num': trial_num,
                    'reach_num': reach_num,
                    'time_sample': timestamp,
                    'x' : values
                }

                rows_0.append(row)

    df0 = pd.DataFrame(rows_0)
    
    for col in time_fields[1:]:
        rows = []
        for trial_num, trial in enumerate(win_df[col]):
            for reach_num, reach in enumerate(trial[0]):
                for timestamp, values in enumerate(reach):
                    row = {
                        col: values
                    }

                    rows.append(row)
        df_col = pd.DataFrame(rows)
        df0[col] = df_col
            
    return df0


def min_max_normalize(vector):
    """
    Normalize a vector between 0 and 1 using Min-Max normalization.

    Args:
    - vector (numpy array): The input vector to be normalized.

    Returns:
    - normalized_vector (numpy array): The normalized vector.
    """
    norm_vec = np.zeros_like(vector)
    for i in range(0, vector.shape[1]):
        min_val = np.min(vector[:,i])
        max_val = np.max(vector[:,i])
        norm_vec[:,i] = (vector[:,i] - min_val) / (max_val - min_val)

    return norm_vec

def train_test_split(df, target_variable = 'x', num_folds = 5):

    """ This function creates disctionnaries to organize the data to perform 
    cross-fold validation. 
    For each fold, we will shuffle the reach trials and use 5 for training, 
    1 for validation, 1 for testing.
    
    Input:
        - df: pandas DataFrame with tidy data
        - target_variable: str, either x, y, z, or angle. Default x.
        - num_folds: (int), number of folds used for cross-validation. Default 5.
    
    Returns:
        - X_train, y_train, X_val, y_val, X_test, y_test: dictionnaries 
            containing the training, validation and testing data for each fold.
        - info_train, info_val, info_test: dictionnaries containing the information 
            for the rain, validation and test data not used for the model, 
            but important for the later data analysis.

    """

    random.seed(42)
    
    trial_ids = np.unique(df['id'].values)
    train_ids = [[] for _ in range(num_folds)] #generic to all folds
    val_ids = [[] for _ in range(num_folds)]  # Create empty lists for each fold
    test_ids = [[] for _ in range(num_folds)]  # Create empty lists for each fold

    num_test = max(int(np.round(0.2*len(trial_ids))),1)
    num_val = max(int(np.round(0.2*(len(trial_ids)-num_test))),1)
    print('Test trials ', num_test)
    print('Val trials', num_val)
    
        
    for i in range(num_folds):
        random.shuffle(trial_ids)
        test_ids[i].extend(trial_ids[-num_test:])
        remaining_ids = [j for j in trial_ids if j not in test_ids[i]]
        train_ids[i].extend(remaining_ids[:-num_val])
        val_ids[i].extend(trial_ids[-num_val:])
        
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    X_val = {}
    y_val = {}
    info_train = {}
    info_val = {}
    info_test = {}

    cols_search = [ 'id','num', 'type', 'KUKAPos', 'trial_num', 'reach_num']
    info_cols = [c for c in cols_search if c in df.columns]

    for fold_idx in range(num_folds):
        print('fold',fold_idx, ' train_ids ',train_ids[fold_idx])
        print('fold',fold_idx, ' val_ids ',val_ids[fold_idx])
        print('fold',fold_idx, ' test_ids ',test_ids[fold_idx])

        df_train = df.loc[df['id'].isin(train_ids[fold_idx])]
        df_val = df.loc[df['id'].isin(val_ids[fold_idx])]
        df_test = df.loc[df['id'].isin(test_ids[fold_idx])]

        train_info = df_train[info_cols]
        X_train_ = np.stack(df_train['both_rates'], axis = 0)
        y_train_ = np.array(df_train[target_variable].tolist())
        val_info = df_val[info_cols]
        X_val_ =  np.stack(df_val['both_rates'], axis = 0)
        y_val_ =  np.array(df_val[target_variable].tolist())
        test_info = df_test[info_cols]
        X_test_ =  np.stack(df_test['both_rates'], axis = 0)
        y_test_ =  np.array(df_test[target_variable].tolist())

        scaler = StandardScaler().fit(X_train_)

        X_train_ = scaler.transform(X_train_)
        X_val_ = scaler.transform(X_val_)
        X_test_ = scaler.transform(X_test_)


        X_train['fold'+str(fold_idx)] = X_train_
        y_train['fold'+str(fold_idx)] = min_max_normalize(y_train_)
        X_val['fold'+str(fold_idx)] = X_val_
        y_val['fold'+str(fold_idx)] = min_max_normalize(y_val_)
        X_test['fold'+str(fold_idx)] = X_test_
        y_test['fold'+str(fold_idx)] = min_max_normalize(y_test_)
        info_train['fold'+str(fold_idx)] = train_info
        info_val['fold'+str(fold_idx)] = val_info
        info_test['fold'+str(fold_idx)] = test_info

    return X_train, y_train, X_val, y_val, X_test, y_test, info_train, info_val, info_test


from collections import Counter

def calculate_mode(data):
    """
    Calculate the mode of a dataset.

    Args:
    - data (list or numpy array): The input dataset.

    Returns:
    - mode: The mode(s) of the dataset.
    """
    # Use Counter to count occurrences of each value
    data = np.round(data, 3)
    counts = Counter(data)

    # Get the maximum count
    max_count = max(counts.values())

    # Find all values with the maximum count (could be multiple modes)
    modes = [value for value, count in counts.items() if count == max_count]

    return modes





    

