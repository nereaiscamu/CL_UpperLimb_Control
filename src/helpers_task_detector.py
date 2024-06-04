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
    return sim_data


def add_offset(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_change = int((ratio/100)*num_total_neurons)
    offsets = np.random.normal(0, 25, size= num_neurons_to_change)
    ind_to_change = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_change)
    for i, offset in zip(ind_to_change, offsets):
        sim_data[:,i] = matrix[:,i] + offset
    return sim_data




# Manage data, models and folders

def save_model(model, task_id, folder):
    # Define the directory path
    models_dir = "../Models"
    task_models_dir = os.path.join(models_dir, folder)

    # Check if the directory exists, if not, create it
    if not os.path.exists(task_models_dir):
        os.makedirs(task_models_dir)

    # Define the file name
    model_file_name = f"Model_Task_{task_id}.pth"  # Use .pth extension for PyTorch models

    # Save the model
    model_path = os.path.join(task_models_dir, model_file_name)

    # Save the model using torch.save
    torch.save(model, model_path)