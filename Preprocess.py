### Imports

import pandas as pd
import numpy as np
import xarray as xr

import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.helpers import *
from sklearn.metrics import *

import pickle
import math

import sys
sys.path.append("c:\\Users\\nerea\\OneDrive\\Documentos\\EPFL_MASTER\\PDM\\Project\\PyalData")
# to change for the actual path where PyalData has been cloned

from pyaldata import *


def preprocess(data_file):
    # Load TrialData .mat file into a DataFrame
    group_df = mat2dataframe(data_file, shift_idx_fields=True, td_name='grp')
    trial_df = mat2dataframe_NC(data_file, shift_idx_fields=True, td_name = 'trial_data') 
    # The main dataframe here is trial_df, the group one will be used 
    # if needed to add some variables to the other one.
    df = trial_df
    # Combine time-bins into longer ones
    td = combine_time_bins(df, 2)
    # Remove low-firing neurons
    td = remove_low_firing_neurons(td, "M1_spikes",  5)
    td = remove_low_firing_neurons(td, "PMd_spikes", 5)
    # Transform signals
    td = transform_signal(td, "M1_spikes",  'sqrt')
    td = transform_signal(td, "PMd_spikes", 'sqrt')
    # Merge signals
    # Signals from the pre-motor and motor cortex are now combines in one same variable, 
    # we do not consider the 2 regions as different but more as functionally working together.
    td = merge_signals(td, ["M1_spikes", "PMd_spikes"], "both_spikes")
    # Compute firing rates
    td = add_firing_rates(td, 'smooth')

    # Here we need to put all end indices together (even if the trial is considered bad)
    # to create the time windows. 
    td['idx_end_complete'] = td.apply(lambda x: add_bad_idx(x['idx_end'], x['idx_bad']), axis=1)
    # Apply the function to each row
    td['bad_indices'] = td.apply(find_bad_indices, axis=1)

    # Merge the 2 tables
    td_all = pd.concat([td,group_df.drop(columns = 'type')], axis = 1)
    cols_to_search = ['index', 'num', 'type', 'tonic_stim_params','KUKAPos',
               'idx_kuka_go', 'idx_reach', 'idx_end_complete',
               'bad_indices', 'x', 'y', 'z', 'angles', 'both_spikes', 'both_rates']
    cols_to_keep = [c for c in cols_to_search if c in td_all.columns]
    td_filt = td_all[cols_to_keep]
    td_filt['test_start'] = td_filt['idx_reach'].apply(lambda x: x[0])
    td_filt['test_end'] = td_filt['idx_end_complete'].apply(lambda x: x[0])

    tidy_df = build_tidy_df(td_filt, start_margin = 1) #before it was 5 but it seemed too much
    tidy_df['target_pos'] = tidy_df.apply(lambda x: np.concatenate([x['x'][4:5], x['y'][4:5], x['z'][4:5]]), axis = 1)
    tidy_df['id'] = tidy_df['trial_num'].astype(str) + '_' + tidy_df['reach_num'].astype(str)

    return tidy_df


if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <date> <folder>")
        sys.exit(1)
    
    # Get the date and folder from command line arguments
    name = sys.argv[1]
    date = sys.argv[2]

    data_dir = "./Data"
    fname = os.path.join(data_dir, "Sansa_2018"+date+".mat")
    tidy_data = preprocess(fname)

    path_to_save_data = os.path.join(data_dir, 'Processed_Data', 'Tidy_'+name+'_'+str(date)+'.pkl')

    # Pickle the data and save it to file
    with open(path_to_save_data, 'wb') as handle:
        pickle.dump(tidy_data, handle, protocol=4)

    print("Saving data...")
