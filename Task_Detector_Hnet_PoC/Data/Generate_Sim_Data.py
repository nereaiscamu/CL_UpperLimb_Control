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

# Setting random seed
random.seed(42)

target_variable = 'vel'

def generate_data(baseline_df, fold):

    size_data = int(baseline_df.shape[0]/2)

    # From those matrices, we will use half the data for one dataset and 
    #the other for a new one. The idea is that the model is exposed to the 
    #other half dataset and recognises the task it has already trained before.

    x_train, y_train, x_val, y_val,\
        x_test, y_test, info_train, info_val,\
            info_test, list_mins_base, \
                list_maxs_base= get_dataset(baseline_df.iloc[:size_data, :], 
                                                fold, 
                                                target_variable= target_variable, 
                                                no_outliers = False, 
                                                force_data = True, 
                                                std = False)

    datasets = {}
    datasets['Data_'+str(0)+'_1'] = x_train, y_train, x_val, y_val, x_test, y_test
    datasets['Data_'+str(0)+'_2'] = x_train, y_train, x_val, y_val, x_test, y_test

    for i in range(1,5):
        data_matrix = np.vstack(baseline_df['both_rates'])
        baseline_df_sim = baseline_df.copy()
        if i == 1:
            sim_data = remove_neurons(data_matrix, 30)
        elif i==2:
            sim_data = shuffle_neurons(data_matrix, 60)
        elif i == 3:
            sim_data = add_gain(data_matrix,50)
        elif i == 4:
            sim_data = add_gain(data_matrix,50)
            
        baseline_df_sim['both_rates'] = sim_data.tolist()
        new_data = baseline_df_sim

        x_train, y_train, x_val, y_val,\
            x_test, y_test, info_train, info_val,\
                info_test, list_mins_base, \
                    list_maxs_base= get_dataset(new_data.iloc[:size_data, :], 
                                                    fold, 
                                                    target_variable= target_variable, 
                                                    no_outliers = False, 
                                                    force_data = True, 
                                                    std = False)
        datasets['Data_'+str(i)+'_1'] = [x_train, y_train, x_val, y_val, x_test, y_test,]

        x_train, y_train, x_val, y_val,\
            x_test, y_test, info_train, info_val,\
                info_test, list_mins_base, \
                    list_maxs_base= get_dataset(new_data.iloc[size_data:, :], 
                                                    fold, 
                                                    target_variable= target_variable, 
                                                    no_outliers = False, 
                                                    force_data = True, 
                                                    std = False)
        datasets['Data_'+str(i)+'_2'] = [x_train, y_train, x_val, y_val, x_test, y_test,]

    # Shuffle the dictionnary keys to check the importance of the task order.
    keys_list = list(datasets.keys())
    random.shuffle(keys_list)
    shuffled_sets = {key: datasets[key] for key in keys_list}

    return shuffled_sets


def main(args):
    name = args.name
    date = args.date
    fold = args.fold
    data_name = args.data_name


    ## Load pre-processed data
    data_path = '../../Data/Processed_Data/Tidy_'+name+'_'+date+'.pkl'

    with open(data_path, 'rb') as file:
        tidy_df = pickle.load(file)
    baseline_df = tidy_df.loc[tidy_df['epoch'] == 'BL']

    sim_data = generate_data(baseline_df, fold)

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

    args = parser.parse_args()
    main(args)