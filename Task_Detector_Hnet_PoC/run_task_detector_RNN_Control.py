### Imports
import pandas as pd
import numpy as np
import pickle
import json
import argparse

# Imports DL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import *
import torch.utils.data as data
from torch.utils.data import Dataset
from hypnettorch.hnets import HyperNetInterface
from hypnettorch.hnets import HMLP
import copy
import time

from helpers_task_detector import *

# Imports from other modules and packages in the project
import os
import sys
# Get the current directory of the script (generate_data.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to reach the grandparent directory (CL Control)
parent_dir = os.path.abspath(os.path.join(current_dir, '..',))
sys.path.append(parent_dir)
print(sys.path)
from src.trainer import *
from src.helpers import *
from src.trainer_hnet import * 
from Models.models import *

#### Model and hyperparameters definition

# Specify that we want our tensors on the GPU and in float32
device = torch.device('cuda:0') #suposed to be cuda
#device = torch.device('cpu') 
dtype = torch.float32

# Set the seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)  # If using CUDA

# Define hyperparameters


results_dict = {}

def run_experiment(experiment, datasets):

    # Retrieve hyperparameters from the experiment_vars dictionary
    hidden_units = experiment['hidden_units']
    num_layers = experiment['num_layers']
    dropout = experiment['dropout']
    lr_detector = experiment['lr_detector']
    lr_step_size = experiment['lr_step_size']
    lr_gamma = experiment['lr_gamma']
    seq_length_LSTM = experiment['seq_length_LSTM']
    batch_size_train = experiment['batch_size_train']
    batch_size_val = experiment['batch_size_val']
    delta = experiment['delta']
    l1_ratio_reg = experiment['l1_ratio_reg']
    alpha_reg = experiment['alpha_reg']
    lr_hnet = experiment['lr_hnet']
    beta_hnet_reg = experiment['beta_hnet_reg']
    thrs = experiment['thrs']
    hidden_layers_hnet = experiment['hidden_layers_hnet']
    emb_size = experiment['embedding_size']

    #### Template x_train and y_train to get the dimensions of the matrices
    for k in datasets.keys():
        break
    num_features = datasets[k][0].shape[2]
    num_dim_output = datasets[k][1].shape[2]

    ### From here all in the loop
   
    for i,s in enumerate(datasets.keys()):
        start_time = time.time()

        task_id = i

        results_dict_subset = {}

        #### Load data
        x_train, y_train, x_val, y_val, x_test, y_test = datasets[s]

        # Define models path and find max_id
        path_RNN_models = './Models/Models_RNN_Control/'+str(experiment['experiment_name'])

            
        results_dict_subset['predicted_task'] = task_id
        results_dict_subset['new_task'] = True

        print('Task_id for this task is ', task_id)

        ####### Define decoder model
        model =  Causal_Simple_RNN(num_features=num_features, 
                    hidden_units= hidden_units, 
                    num_layers = num_layers, 
                    out_dims = num_dim_output,
                    dropout = dropout).to(device)

        
        # Training the task detector model
        train_losses, val_losses = \
            train_model(model, 
                        x_train, 
                        y_train, 
                        x_val, 
                        y_val,
                        lr=  lr_detector,
                        lr_step_size=lr_step_size,
                        lr_gamma= lr_gamma,
                        sequence_length_LSTM=seq_length_LSTM,
                        batch_size_train = batch_size_train,
                        batch_size_val = batch_size_val,
                        num_epochs=1000, 
                        delta = delta,                 
                        regularizer= regularizer,
                        layer_type = 'rnn', 
                        l1_ratio = l1_ratio_reg,
                        alpha = alpha_reg,     
                        early_stop = 5)
        
        print('Train losses', train_losses)
        results_dict_subset['hnet_train_losses'] = train_losses
        results_dict_subset['hnet_val_losses'] = val_losses
        
        # Evaluate model on first seen data
        y_hat, y_true, train_score, v_score, test_score = eval_model( x_train, y_train,
                                                                    x_val, y_val,
                                                                    x_test, y_test, 
                                                                    model, 
                                                                    metric = 'r2')
        results_dict_subset['y_true_hnet'] = y_true
        results_dict_subset['y_pred_hnet'] = y_hat
        results_dict_subset['r2_test_hnet'] = test_score

        print('R2 for the task', task_id, ' is ', v_score)

        if v_score <thrs:
            print('WARNING, THE TASK COULD NOT BE LEARNED BY THE DETECTOR')
            #break
        else:
            print('Task learned without issues.')
        # Save the trained model
        save_model(model, task_id, path_RNN_models)
  
        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        results_dict_subset['training_time'] = elapsed_time

        results_dict[s] = results_dict_subset

    return results_dict



def main(args):

    index = args.index

    # Load the list of experiments from JSON
    with open(os.path.join('config.json'), 'r') as f:
        experiments = json.load(f)

    if index == -1:
        for exp in range(44,50):
            experiment = experiments[exp]
            name = experiment['experiment_name']
            print('Running esperiment ', name, '_control')

            # Loading data
            data = experiment['data']
            data_dir = "./Data/"
            with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
                sets = pickle.load(fp)

            results_dict = run_experiment(experiment, sets)

            path_to_results = os.path.join('.','Results')

            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results)

            file_path = os.path.join(path_to_results, name+'_RNN_control.pkl')
            
            # Save the dictionary to a file usnig pickle
            with open(file_path, 'wb') as fp:
                pickle.dump(results_dict, fp)
    else:
        experiment = experiments[index]
        name = experiment['experiment_name']
        print('Running esperiment ', name, '_control')

        # Loading data
        data = experiment['data']
        data_dir = "./Data/"
        with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
            sets = pickle.load(fp)

        results_dict = run_experiment(experiment, sets)

        path_to_results = os.path.join('.','Results')

        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        file_path = os.path.join(path_to_results, name+'_RNN_control.pkl')
        
        # Save the dictionary to a file usnig pickle
        with open(file_path, 'wb') as fp:
            pickle.dump(results_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Main script to run experiments" 
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index to iterate over the dictionary",
    )

    args = parser.parse_args()
    main(args)

