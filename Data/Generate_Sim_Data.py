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



def generate_sim_data(data_matrix, num_sets):

    """ Function to generate the simulated data as a perturbation of the original 
    neural data using 4 different approaches:
        - Increasing the firing rate of some neurons by 20%
        - Decreasing the firing rate of some random neurons by 20%
        - Replacing some neurons by some others from the dataset
        - Putting some firing rate trains to 0 (we don't remove neurons for now to keep the same matrix dimensions
        
    Inputs:
        - data_matrix: np.array containing all contatenated trials over time (rows) for each neuron (columns)
        - num_sets: integer, number of simulated matrices to generate
    
    Output: 
        - simulated_data: dictionnary including all sets of simulated data"""

    num_neurons = data_matrix.shape[1]
    num_samples = data_matrix.shape[0]
    num_sets = num_sets
    simulated_data = {}
        
    for set_ in range(num_sets):
        new_data = data_matrix.copy()

        num_fr_gain = int(np.ceil((random.sample(list(np.arange(10,20)),1)[0]/100)*num_samples))
        idx_fr_gain = random.sample(list(np.arange(0,num_samples)), num_fr_gain)
        for i in idx_fr_gain:
            new_data[i,:] = new_data[i,:]*1.2

        num_fr_loss = int(np.ceil((random.sample(list(np.arange(10,20)),1)[0]/100)*num_samples))
        idx_fr_loss = random.sample(list(np.arange(0,num_samples)), num_fr_loss)
        for i in idx_fr_loss:
            new_data[i,:] = new_data[i,:]*0.8

        num_replaced = int(np.ceil((random.sample(list(np.arange(0,10)),1)[0]/100)*num_neurons))
        idx_replaced = random.sample(list(np.arange(0,num_neurons)), num_replaced)
        neurons_to_replace = [i for i in np.arange(0,num_neurons) if i not in idx_replaced]
        idx_to_replace = random.sample(neurons_to_replace, num_replaced)
        for old,new in zip(idx_replaced, idx_to_replace):
            new_data[:,old] = new_data[:,new]

        original_neurons = [i for i in np.arange(0,num_neurons) if i not in idx_replaced]
        
        num_removed = int(np.ceil((random.sample(list(np.arange(20,30)),1)[0]/100)*num_neurons))
        idx_removed = random.sample(original_neurons, num_removed)
        for i in idx_removed:
            new_data[:,i] = 0
        #new_data = np.delete(new_data, idx_removed, axis = 1) --> If I remove data i will not be able to use the same models from one to another. Problem to solve later.
    
        simulated_data['Set_'+str(set_)] = new_data
    
    return simulated_data




if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <name> <date> <num_generated_sets>")
        sys.exit(1)

    # Get the date and folder from command line arguments
    name = sys.argv[1]
    date = sys.argv[2]
    num_generated_sets = int(sys.argv[3])


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


    sim_data = generate_sim_data(data_matrix, num_sets = num_generated_sets)

    path_to_save_data = os.path.join(data_dir, 'Simulated_'+str(num_generated_sets)+'_'+name+'_'+str(date)+'.pkl')

    # Pickle the data and save it to file
    with open(path_to_save_data, 'wb') as handle:
        pickle.dump(sim_data, handle, protocol=4)

    print("Saving data...")