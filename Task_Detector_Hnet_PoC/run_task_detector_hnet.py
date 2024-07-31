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


    ####### Define task detector model
    task_detector_model =  Causal_Simple_RNN(num_features=num_features, 
                    hidden_units= hidden_units, 
                    num_layers = num_layers, 
                    out_dims = num_dim_output,
                    dropout = dropout).to(device)
    

    #### Defining the template, main and hnet models and initializing them
    
    # We can use the task detector as a template for the main model
    param_shapes = [p.shape for p in list(task_detector_model.parameters())]

    num_conditions = 60 # we want more possible conditions than what we can reach
    size_task_embedding = 24 # changed to test on 22/07 it was 8 # seemed to work well 

    hnet = HMLP(param_shapes, uncond_in_size=0,
                cond_in_size=emb_size, #usually we had size_task_embedding
                layers=hidden_layers_hnet, # trying different values on 22/07, before that it was [13]
                num_cond_embs=num_conditions).to(device)

    for param in hnet.parameters():
        param.requires_grad = True

    hnet.apply_hyperfan_init()

    w_test = hnet(cond_id = 0)

    LSTM_ = False

    model = RNN_Main_Model(num_features= num_features, hnet_output = w_test,  hidden_size = hidden_units,
                                num_layers= num_layers,out_dims=num_dim_output,  
                                dropout= dropout,  LSTM_ = LSTM_).to(device)
    # Make sure the parameters of the model do not require gradient, as we want only to learn the hnet params
    for param in model.parameters():
        param.requires_grad = False

    
    ### From here all in the loop
  
    calc_reg = False
   
    results_dict = {}

    for s in datasets.keys():

        results_dict_subset = {}

        #### Load data
        x_train, y_train, x_val, y_val, x_test, y_test = datasets[s]

        # Define models path and find max_id
        path_recog_models = './Models/Models_Task_Recognition/'+str(experiment['experiment_name'])
        path_hnet_models = './Models/Models_HNET/'+str(experiment['experiment_name'])
        # Check if the directory exists, if not, create it
        if not os.path.exists(path_recog_models):
            os.makedirs(path_recog_models)

        trained_detectors = np.sort(os.listdir(path_recog_models))

        r2_list = []
        
        for i,m in enumerate(trained_detectors):
            model_i = torch.load(os.path.join(path_recog_models, m)).to(device)
            model_i.eval()
            _, _, _, r2_i,_ = eval_model(x_train, 
                                        y_train, 
                                        x_val, 
                                        y_val,
                                        x_test, 
                                        y_test,
                                        model_i, 
                                        metric = 'r2')
            r2_list.append(r2_i)

        if not r2_list:
            max_id = 0
            task_id = 0
            
            results_dict_subset['predicted_task'] = task_id
            results_dict_subset['new_task'] = True

            print('Training on the first task!')
            print('Task_id for this task is ', task_id)

            #Define the task detector model
            task_detector_i =  Causal_Simple_RNN(num_features=num_features, 
                        hidden_units= hidden_units, 
                        num_layers = num_layers, 
                        out_dims = num_dim_output,
                        dropout = dropout).to(device)
            
            # Training the task detector model
            train_losses, val_losses = \
                train_model(task_detector_i, 
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
                            regularizer= Regularizer_RNN, 
                            l1_ratio = l1_ratio_reg,
                            alpha = alpha_reg,     
                            early_stop = 5)
            results_dict_subset['detector_train_losses'] = train_losses
            results_dict_subset['detector_val_losses'] = val_losses
            
            # Evaluate model on first seen data
            y_hat, y_true,train_score, v_score, test_score = eval_model( x_train, y_train,
                                                                        x_val, y_val,
                                                                        x_test, y_test, 
                                                                        task_detector_i, 
                                                                        metric = 'r2')
            results_dict_subset['y_true_detector'] = y_true
            results_dict_subset['y_pred_detector'] = y_hat
            results_dict_subset['r2_test_detector'] = test_score

            print('R2 for the task', task_id, ' is ', v_score)

            if v_score <thrs:
                print('WARNING, THE TASK COULD NOT BE LEARNED BY THE DETECTOR')
                #break
            else:
                print('Task learned without issues.')
            # Save the trained model
            save_model(task_detector_i, task_id, path_recog_models)
            print('Training now on the hnet')
            # Record the start time
            start_time = time.time()
            train_losses_, val_losses_, best_w_ =train_current_task(
                                                                model, 
                                                                hnet,
                                                                y_train, 
                                                                x_train, 
                                                                y_val,
                                                                x_val, 
                                                                calc_reg = calc_reg,
                                                                cond_id = int(task_id),
                                                                lr= lr_hnet,
                                                                lr_step_size= 5,
                                                                lr_gamma= lr_gamma, #0.9
                                                                sequence_length_LSTM = seq_length_LSTM, #15
                                                                batch_size_train = batch_size_train, #15
                                                                batch_size_val = batch_size_train, #15
                                                                num_epochs= 1000, 
                                                                delta = delta,
                                                                beta = beta_hnet_reg, 
                                                                regularizer= reg_hnet,
                                                                l1_ratio = l1_ratio_reg, #0.5
                                                                alpha = 0.01, # before it was alpha_reg. Changed in 22/07.
                                                                early_stop = 5,
                                                                chunks = False)
            W_best = hnet(cond_id = task_id)
            r2, _ = calc_explained_variance_mnet(x_val, y_val, W_best, model)
            r2_test, y_pred_test = calc_explained_variance_mnet(x_test, y_test, W_best, model)
            results_dict_subset['y_true_hnet'] = y_test
            results_dict_subset['y_pred_hnet'] = y_pred_test
            results_dict_subset['r2_test_hnet'] = r2_test
            print('R2 for the HNET on Task ', task_id, ' is ', r2)
            # Save the trained model
            save_model(hnet, task_id, path_hnet_models)
            results_dict_subset['hnet_train_losses'] = train_losses_
            results_dict_subset['hnet_val_losses'] = val_losses_
            # Record the end time
            end_time = time.time()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            results_dict_subset['training_time'] = elapsed_time
                
        else:
            max_id = len(trained_detectors) - 1
            max_r2 = max(r2_list)


            if max_r2 > thrs:

                # Show performance on the hnet
                print('This data comes from a known task. ')
                task_id = np.argmax(r2_list)
                results_dict_subset['predicted_task'] = task_id
                results_dict_subset['new_task'] = False
                results_dict_subset['r2_test_detector'] = max_r2
                print('Task_id for this task is ', task_id)
                W_i = hnet(cond_id = int(task_id))
                r2, y_pred_val = calc_explained_variance_mnet(x_val, y_val, W_i, model)
                r2_test, y_pred_test = calc_explained_variance_mnet(x_test, y_test, W_i, model)
                results_dict_subset['r2_test_hnet'] = r2_test
                results_dict_subset['y_true_hnet'] = y_test
                results_dict_subset['y_pred_hnet'] = y_pred_test
                print('R2 for the HNET on task', task_id, ' is ', r2)
                

            else:
                
                print('This data comes from a different task !')
                max_id += 1
                print('max id has changed to ', max_id)
                task_id = max_id

                if task_id >0:
                    calc_reg = True # --> CORRECTED BUG.BEFORE THIS WAS DEFINED BEFORE DEFINING THE NEW TASK ID.

                results_dict_subset['predicted_task'] = task_id
                results_dict_subset['new_task'] = True
                print('Task_id for this task is ', task_id)
                task_detector_i =  Causal_Simple_RNN(num_features=num_features, 
                            hidden_units= hidden_units, 
                            num_layers = num_layers, 
                            out_dims = num_dim_output,
                            dropout = dropout).to(device)

                # Training the task detector model
                train_losses, val_losses = \
                    train_model(task_detector_i, 
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
                                regularizer= Regularizer_RNN, 
                                l1_ratio = l1_ratio_reg,
                                alpha = alpha_reg,     
                                early_stop = 5)
                
                results_dict_subset['detector_train_losses'] = train_losses
                results_dict_subset['detector_val_losses'] = val_losses
                # Evaluate model on first seen data
                y_hat, y_true,train_score, v_score, test_score = eval_model( x_train, y_train,
                                                                            x_val, y_val,
                                                                            x_test, y_test, 
                                                                            task_detector_i, 
                                                                            metric = 'r2')
                results_dict_subset['y_true_detector'] = y_true
                results_dict_subset['y_pred_detector'] = y_hat
                results_dict_subset['r2_test_detector'] = test_score
                print('R2 for the task', task_id, ' is ', v_score)

                if v_score <thrs:
                    print('WARNING, THE TASK COULD NOT BE LEARNED BY THE DETECTOR')
                    #break
                else:
                    print('Task learned without issues.')

                # Save the trained model
                save_model(task_detector_i, task_id, path_recog_models)
                print('Training now on the hnet')
                # Record the start time
                start_time = time.time()
                train_losses_, val_losses_, best_w_ =train_current_task(
                                                                    model, 
                                                                    hnet,
                                                                    y_train, 
                                                                    x_train, 
                                                                    y_val,
                                                                    x_val, 
                                                                    calc_reg = calc_reg,
                                                                    cond_id = int(task_id),
                                                                    lr=lr_hnet,
                                                                    lr_step_size=5,
                                                                    lr_gamma= lr_gamma, #0.9
                                                                    sequence_length_LSTM = seq_length_LSTM, #15
                                                                    batch_size_train = batch_size_train, #15
                                                                    batch_size_val = batch_size_train, #15
                                                                    num_epochs=1000, 
                                                                    delta = delta,
                                                                    beta = beta_hnet_reg,             
                                                                    regularizer=reg_hnet,# None, --> changed on 20/07/24 to check if learning is faster.
                                                                    l1_ratio = l1_ratio_reg, #0.5
                                                                    alpha = alpha_reg,    
                                                                    early_stop = 5,
                                                                    chunks = False)
                results_dict_subset['hnet_train_losses'] = train_losses_
                results_dict_subset['hnet_val_losses'] = val_losses_
                # Record the end time
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                results_dict_subset['training_time'] = elapsed_time


                W_best = hnet(cond_id = task_id)
                r2, _ = calc_explained_variance_mnet(x_val, y_val, W_best, model)
                r2_test, y_pred_test = calc_explained_variance_mnet(x_test, y_test, W_best, model)
                results_dict_subset['y_true_hnet'] = y_test
                results_dict_subset['y_pred_hnet'] = y_pred_test
                results_dict_subset['r2_test_hnet'] = r2_test
                print('R2 for the HNET on Task ', task_id, ' is ', r2)
                # Save the trained model
                save_model(hnet, task_id, path_hnet_models)

        results_dict[s] = results_dict_subset

    return results_dict



def main(args):

    index = args.index
    sort = bool(args.sort)

    # Load the list of experiments from JSON
    with open(os.path.join('config.json'), 'r') as f:
        experiments = json.load(f)

    if index == -1:
        for exp in range(125,128): 
            experiment = experiments[exp]
            name = experiment['experiment_name']
            print('Running esperiment ', name)

            # Loading data
            data = experiment['data']
            data_dir = "./Data/"
            with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
                sets = pickle.load(fp)
            print('Data found')

            if sort:
                print('Sorting the data')
                num_trials = experiment['num_trials']
                # Either keep only a number of trials from the dataset or make sure baseline is the first task
                sets = create_sets(sets, num_trials)
                # Save the data to understand which experiment was run
                path_to_save_data = os.path.join(data_dir, data+'_'+str(num_trials)+'trials.pkl')
                # Pickle the data and save it to file
                with open(path_to_save_data, 'wb') as handle:
                    pickle.dump(sets, handle, protocol=4)

                print("Saving data...")
  
            # Now running experiment on the desired trial number
            print('Running experiment...')
            results_dict = run_experiment(experiment, sets)

            path_to_results = os.path.join('.','Results')

            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results)

            file_path = os.path.join(path_to_results, name+'.pkl')
            
            # Save the dictionary to a file usnig pickle
            with open(file_path, 'wb') as fp:
                pickle.dump(results_dict, fp)
    else:
        experiment = experiments[index]
        name = experiment['experiment_name']
        print('Running esperiment ', name)

        # Loading data
        data = experiment['data']
        data_dir = "./Data/"
        with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
            sets = pickle.load(fp)

        if sort:
                print('Sorting the data')
                num_trials = experiment['num_trials']
                # Either keep only a number of trials from the dataset or make sure baseline is the first task
                sets = create_sets(sets, num_trials)
                # Save the data to understand which experiment was run
                path_to_save_data = os.path.join(data_dir, data+'_'+str(num_trials)+'trials.pkl')
                # Pickle the data and save it to file
                with open(path_to_save_data, 'wb') as handle:
                    pickle.dump(sets, handle, protocol=4)

                print("Saving data...")

        # Now running experiment on the desired trial number
        print('Running experiment...')
        results_dict = run_experiment(experiment, sets)

        path_to_results = os.path.join('.','Results')

        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        file_path = os.path.join(path_to_results, name+'.pkl')
        
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

    parser.add_argument(
        "--sort",
        type=int,
        default=0,
        help="If data needs to be sorted to get the baseline first",
    )

    args = parser.parse_args()
    main(args)

