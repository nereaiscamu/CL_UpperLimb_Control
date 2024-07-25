import random
import numpy as np
import os
import sys
import torch

# Generate simulated perturbed neural data

def remove_neurons(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_removed = int((ratio/100)*num_total_neurons)
    idx_removed = random.sample(list(np.arange(0,num_total_neurons)), num_removed)
    for i in idx_removed:
        sim_data[:,i] = 0
    return sim_data


def shuffle_neurons(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_shuffle = int((ratio/100)*num_total_neurons)
    ind_to_permute = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_shuffle)
    ind_to_permute = np.sort(ind_to_permute)
    permuted_indices = np.random.permutation(ind_to_permute)
    for i, new_i in zip(ind_to_permute, permuted_indices):
        sim_data[:,i] = matrix[:,new_i]
    return sim_data


def add_gain(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_change = int((ratio/100)*num_total_neurons)
    gains = np.random.normal(1, 2, size= num_neurons_to_change)
    ind_to_change = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_change)
    for i, gain in zip(ind_to_change, gains):
        sim_data[:,i] = matrix[:,i]*gain
    return abs(sim_data) # --> changed to abs as FR can't be negative. 18/07/2024


def add_offset(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_change = int((ratio/100)*num_total_neurons)
    offsets = np.random.normal(0, 25, size= num_neurons_to_change)
    ind_to_change = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_change)
    for i, offset in zip(ind_to_change, offsets):
        sim_data[:,i] = matrix[:,i] + offset
    return abs(sim_data) # --> changed to abs as FR can't be negative. 18/07/2024





# Manage data, models and folders

def save_model(model, task_id, path):

    # Check if the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Define the file name
    model_file_name = f"Model_Task_{task_id}.pth"  # Use .pth extension for PyTorch models

    # Save the model
    model_path = os.path.join(path, model_file_name)

    # Save the model using torch.save
    torch.save(model, model_path)


# Creating smaller datasets depending on the trial number
def get_reduced_sets(data, num_trials = -1):
    
    if num_trials == -1:
        new_data = data
        
    else:
        new_data = {}

        for s in data.keys():
            x_train, y_train, x_val, y_val, x_test, y_test = data[s]
            trials_train = []
            trials_val = []
            trials_test = []        
            num_test_trials = min(10, num_trials)

            for i in range(num_trials):
                random.seed()
                trials_train.append(random.randint(0,x_train.shape[0]-1))

            for i in range(num_test_trials):
                random.seed()
                trials_val.append(random.randint(0,x_val.shape[0]-1))
                trials_test.append(random.randint(0,x_test.shape[0]-1))

                x_train_reduced = np.array([x_train[i,:,:] for i in trials_train])
                y_train_reduced = np.array([y_train[i,:,:] for i in trials_train])
                x_val_reduced = np.array([x_val[i,:,:] for i in trials_val])
                y_val_reduced = np.array([y_val[i,:,:] for i in trials_val])
                x_test_reduced = np.array([x_test[i,:,:] for i in trials_test])
                y_test_reduced = np.array([y_test[i,:,:] for i in trials_test])
            
            new_data[s] = [x_train_reduced,
                            y_train_reduced,
                            x_val_reduced,
                            y_val_reduced,
                            x_test_reduced,
                            y_test_reduced]
    return new_data

# Ensuring the first learned task is Baseline. 
def ensure_baseline_first(d):
    keys = list(d.keys())
    if keys[0] != 'Data_0_1':
        keys.remove('Data_0_1')
        keys.insert(0, 'Data_0_1')
    updated_dict = {k: d[k] for k in keys}
    return updated_dict

# Shuffle datasets

def shuffle_sets(datasets):
# Shuffle the dictionnary keys to check the importance of the task order.
    keys_list = list(datasets.keys())
    random.seed()
    random.shuffle(keys_list)
    shuffled_sets = {key: datasets[key] for key in keys_list}
    return shuffled_sets