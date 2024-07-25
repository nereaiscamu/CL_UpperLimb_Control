""" Simulating Neural Datasets

This script is meant to use the variable "both_rates", containing the firing rates of 
both M1 and PdM areas and generate a number X of simulations of this same variable 
containing different perturbations"


"""
## Imports
### Imports
import pandas as pd
import numpy as np
import pickle
import argparse
import math

# Imports from other modules and packages in the project
import os
import sys
# Get the current directory of the script (generate_data.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the grandparent directory (CL Control)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..','..',))

# Add the grandparent directory to sys.path
sys.path.append(grandparent_dir)
from src.helpers import *
from Task_Detector_Hnet_PoC.helpers_task_detector import *

import time

# Get the current time as an integer
current_time = int(time.time())

# Set the random seed using the current time
random.seed(current_time)

target_variable = 'vel'

def get_reduced_sets(data, fold, num_trials = -1):
    x_train, y_train, x_val, y_val,\
    x_test, y_test, info_train, info_val,\
        info_test, list_mins_base, \
            list_maxs_base= get_dataset(data, 
                                            fold, 
                                            target_variable= target_variable, 
                                            no_outliers = False, 
                                            force_data = True, 
                                            std = False)
    if num_trials == -1:

        size_train = int(x_train.shape[0]/2)
        size_val = int(x_val.shape[0]/2)
        size_test = int(x_test.shape[0]/2)

        data1 = [x_train[:size_train,:,:],
                y_train[:size_train,:,:],
                x_val[:size_val,:,:],
                y_val[:size_val,:,:],
                x_test[:size_test,:,:],
                y_test[:size_test,:,:]]

        data2 = [x_train[size_train:,:,:],
                y_train[size_train:,:,:],
                x_val[size_val:,:,:],
                y_val[size_val:,:,:],
                x_test[size_test:,:,:],
                y_test[size_test:,:,:]]
    else:
        trials_train = []
        trials_val = []
        trials_test = []
        
        num_test_trials = min(10, num_trials)

        for i in range(num_trials*2):
            random.seed()
            trials_train.append(random.randint(0,x_train.shape[0]-1))

        for i in range(num_test_trials*2):
            random.seed()
            trials_val.append(random.randint(0,x_val.shape[0]-1))
            trials_test.append(random.randint(0,x_test.shape[0]-1))

        x_train_reduced = np.array([x_train[i,:,:] for i in trials_train])
        y_train_reduced = np.array([y_train[i,:,:] for i in trials_train])
        x_val_reduced = np.array([x_val[i,:,:] for i in trials_val])
        y_val_reduced = np.array([y_val[i,:,:] for i in trials_val])
        x_test_reduced = np.array([x_test[i,:,:] for i in trials_test])
        y_test_reduced = np.array([y_test[i,:,:] for i in trials_test])


        data1 =  [x_train_reduced[:num_trials,:,:],
                y_train_reduced[:num_trials,:,:],
                x_val_reduced[:num_test_trials,:,:],
                y_val_reduced[:num_test_trials,:,:],
                x_test_reduced[:num_test_trials,:,:],
                y_test_reduced[:num_test_trials,:,:]]

        data2 = [x_train_reduced[num_trials:,:,:],
                y_train_reduced[num_trials:,:,:],
                x_val_reduced[num_test_trials:,:,:],
                y_val_reduced[num_test_trials:,:,:],
                x_test_reduced[num_test_trials:,:,:],
                y_test_reduced[num_test_trials:,:,:]]
        
    return data1, data2


def generate_data(data, fold, num_trials):

    # From those matrices, we will use half the data for one dataset and 
    #the other for a new one. The idea is that the model is exposed to the 
    #other half dataset and recognises the task it has already trained before.

    datasets = {}

    datasets['Data_'+str(0)+'_1'], datasets['Data_'+str(0)+'_2'] = get_reduced_sets(data, fold, num_trials = num_trials)

    
    for i in range(1,5):
        data_matrix = np.vstack(data['both_rates'])
        baseline_df_sim = data.copy()
        if i == 1:
            sim_data = remove_neurons(data_matrix, 30)
            sim_data = shuffle_neurons(sim_data,40)
        elif i==2:
            sim_data = shuffle_neurons(data_matrix, 50)
        elif i == 3:
            sim_data = add_gain(data_matrix,60)
        elif i == 4:
            sim_data = add_offset(data_matrix,60)
            
        baseline_df_sim['both_rates'] = sim_data.tolist()
        new_data = baseline_df_sim

        datasets['Data_'+str(i)+'_1'],datasets['Data_'+str(i)+'_2'] = get_reduced_sets(new_data, fold, num_trials = num_trials)

    # Shuffle the dictionnary keys to check the importance of the task order.
    keys_list = list(datasets.keys())
    random.seed()
    random.shuffle(keys_list)
    shuffled_sets = {key: datasets[key] for key in keys_list}

    return shuffled_sets


def main(args):
    name = args.name
    date = args.date
    fold = args.fold
    data_name = args.data_name
    num_trials = args.num_trials


    ## Load pre-processed data
    data_path = '../../Data/Processed_Data/Tidy_'+name+'_'+date+'.pkl'

    with open(data_path, 'rb') as file:
        tidy_df = pickle.load(file)
        
    baseline_df = tidy_df.loc[tidy_df['epoch'] == 'BL']

    sim_data = generate_data(baseline_df, fold, num_trials)

    data_dir = './'
    path_to_save_data = os.path.join(data_dir, 'Sim_Data_'+data_name+'.pkl')

    # Pickle the data and save it to file
    with open(path_to_save_data, 'wb') as handle:
        pickle.dump(sim_data, handle, protocol=4)

    print("Saving data...")

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    parser = argparse.ArgumentParser(
    description="Main script to run experiments"
    )

    parser.add_argument(
        "--name",
        type=str,
        default='Chewie',
        help="Name of the participant from whom the data was recorded",
    )
    parser.add_argument(
        "--date",
        type=str,
        default='1007',
        help="Date when the data was recorded",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Data fold to use (from 0 to 4)",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default='Test',
        help="How the data should be saved, usually name of the experiment",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=-1,
        help="Number of trials to include for the experiment. Default -1,means we use all trials.",
    )

    args = parser.parse_args()
    main(args)