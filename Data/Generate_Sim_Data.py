""" Simulating Neural Datasets

This script is meant to use the variable "both_rates", containing the firing rates of 
both M1 and PdM areas and generate a number X of simulations of this same variable 
containing different perturbations"


"""
## Imports
import os
import sys
import pandas as pd
import pickle
import numpy as np
import random

# Setting random seed
random.seed(42)


def shuffle_columns(matrix, ratio):
    new_matrix = matrix.copy()
    # Generate a random permutation of column indices
    num_neurons = matrix.shape[1]
    num_columns_to_shuffle = int(ratio*num_neurons)
    ind_to_permute = random.sample(list(np.arange(0,num_neurons)), num_columns_to_shuffle)
    ind_to_permute = np.sort(ind_to_permute)
    permuted_indices = np.random.permutation(ind_to_permute)
    for i, new_i in zip(ind_to_permute, permuted_indices):
        new_matrix[:,i] = matrix[:,new_i]

    return new_matrix, ind_to_permute


def generate_sim_data(data_matrix, num_sets, ratio):

    """ Function to generate the simulated data as a perturbation of the original 
    neural data using 4 different approaches:
        - Multiplying fire rate of neurons by gain from gaussian distribution.
        - Adding offsets to the neurons sampled from gaussian distribution.
        - Shuffling some neurons from the dataset and keeping the rest where they are.
        - Putting some firing rate trains to 0 (we don't remove neurons for now to keep the same matrix dimensions.
        
    Inputs:
        - data_matrix: np.array containing all contatenated trials over time (rows) for each neuron (columns)
        - num_sets: integer, number of simulated matrices to generate
        - ratio: int, num from 0 to 100 used to know the percentage of neurons to remove/shuffle
    
    Output: 
        - simulated_data: dictionnary including all sets of simulated data"""

    num_neurons = data_matrix.shape[1]
    num_samples = data_matrix.shape[0]
    num_sets = num_sets
    simulated_data = {}
        
    for set_ in range(num_sets):
        new_data = data_matrix.copy()

        ### a) Multiplying each column by a random gain from gaussian dist.
        gains = np.random.normal(1, 0.3, size=new_data.shape[1])
        # Multiply each column of the matrix by the corresponding gain value
        new_data = new_data * gains[:, np.newaxis].T

        ### b) Shuffle neuron positions
        new_data, ind_to_permute = shuffle_columns(new_data, ratio)

        original_neurons = [i for i in np.arange(0,num_neurons) if i not in ind_to_permute]
        
        ### c) Remove neurons
        num_removed = int(ratio*num_neurons)
        idx_removed = random.sample(original_neurons, num_removed)
        for i in idx_removed:
            new_data[:,i] = 0
        
         ### d) Random offsets
        offsets = np.random.normal(0, 0.3, size=new_data.shape[1])
        # Multiply each column of the matrix by the corresponding gain value
        new_data = new_data + offsets[:, np.newaxis].T
    
        simulated_data['Set_'+str(set_)] = new_data
    
    return simulated_data




if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python script_name.py <name> <date> <num_generated_sets> <ratio>")
        sys.exit(1)

    # Get the date and folder from command line arguments
    name = sys.argv[1]
    date = sys.argv[2]
    num_generated_sets = int(sys.argv[3])
    ratio = int(sys.argv[4])/100


    data_dir = "./Processed_Data"
    dataset = os.path.join(data_dir, "Tidy_"+name+'_'+date+".pkl")

    with open(dataset, 'rb') as file:
        tidy_df = pickle.load(file)

    baseline_df = tidy_df.loc[tidy_df['epoch'] == 'BL']
    baseline_df = baseline_df [['index', 'monkey', 'date', 'task', 'target_direction', 'id',
        'result', 'bin_size', 'perturbation', 'perturbation_info', 'epoch',
        'idx_trial_start', 'idx_target_on', 'idx_go_cue', 'idx_movement_on',
        'idx_peak_speed', 'idx_trial_end', 'pos', 'vel', 'acc', 'force',
        'M1_spikes', 'M1_unit_guide', 'PMd_spikes', 'PMd_unit_guide',
        'both_spikes', 'M1_rates', 'PMd_rates', 'both_rates',]]


    data_matrix = np.vstack(baseline_df['both_rates'])


    sim_data = generate_sim_data(data_matrix, num_sets = num_generated_sets, ratio = ratio)

    path_to_save_data = os.path.join(data_dir, 'Simulated_'+str(num_generated_sets)+'_'+'ratio'+str(ratio)+'_'+name+'_'+str(date)+'.pkl')

    # Pickle the data and save it to file
    with open(path_to_save_data, 'wb') as handle:
        pickle.dump(sim_data, handle, protocol=4)

    print("Saving data...")