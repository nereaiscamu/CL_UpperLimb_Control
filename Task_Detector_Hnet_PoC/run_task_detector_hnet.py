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

# Imports from other modules and packages in the project
import os
import sys
sys.path.append('../')
from src.helpers import *
from src.trainer import *
from src.trainer_hnet import * 
from src.helpers_task_detector import *
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


results_dict = []

def run_experiment(experiment, datasets):

    # Experiment has all information about hyperparams etc.
    for key, value in experiment.items():
        exec(f"{key} = experiment['{key}']")

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
    size_task_embedding = 8 # seemed to work well 

    hnet = HMLP(param_shapes, uncond_in_size=0,
                cond_in_size=size_task_embedding,
                layers=[13], 
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
    thrs = 0.8
    calc_reg = False
    predicted_tasks = []

    for s in datasets.keys():

        #### Load data
        x_train, y_train, x_val, y_val, x_test, y_test = datasets[s]

        # Define models path and find max_id
        path_recog_models = './Models/Models_Task_Recognition/'+str(experiment['experiment_name'])
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
            predicted_tasks.append([s,task_id])

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
            
            # Evaluate model on first seen data
            y_hat, y_true,train_score, v_score, test_score = eval_model( x_train, y_train,
                                                                        x_val, y_val,
                                                                        x_test, y_test, 
                                                                        task_detector_i, 
                                                                        metric = 'r2')
            print('R2 for the task', task_id, ' is ', v_score)

            if v_score <thrs:
                print('ERROR, THE TASK COULD NOT BE LEARNED BY THE DETECTOR')
                break
            else:
                print('Task learned without issues.')
            # Save the trained model
            save_model(task_detector_i, task_id, 'Models_Task_Recognition/'+str(experiment['experiment_name']))
            print('Training now on the hnet')
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
                                                                regularizer=reg_hnet,
                                                                l1_ratio = l1_ratio_reg, #0.5
                                                                alpha = alpha_reg,    
                                                                early_stop = 5,
                                                                chunks = False)
            W_best = hnet(cond_id = task_id)
            r2 = calc_explained_variance_mnet(x_val, y_val, W_best, model)
            print('R2 for the HNET on Task ', task_id, ' is ', r2)
            # Save the trained model
            save_model(hnet, task_id, "HNET_Task_Recog/"+str(experiment['experiment_name']))
                
        else:
            max_id = len(trained_detectors) - 1
            max_r2 = max(r2_list)

            if max_r2 > thrs:

                # Show performance on the hnet
                print('This data comes from a known task. ')
                task_id = np.argmax(r2_list)
                predicted_tasks.append([s,task_id])
                print('Task_id for this task is ', task_id)
                W_i = hnet(cond_id = int(task_id))
                r2 = calc_explained_variance_mnet(x_val, y_val, W_i, model)
                print('R2 for the HNET on task', task_id, ' is ', r2)

            else:
                if task_id >0:
                    calc_reg = True
                print('This data comes from a different task !')
                max_id += 1
                print('max id has changed to ', max_id)
                task_id = max_id
                predicted_tasks.append([s,task_id])
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
                # Evaluate model on first seen data
                y_hat, y_true,train_score, v_score, test_score = eval_model( x_train, y_train,
                                                                            x_val, y_val,
                                                                            x_test, y_test, 
                                                                            task_detector_i, 
                                                                            metric = 'r2')
                print('R2 for the task', task_id, ' is ', v_score)

                if v_score <thrs:
                    print('ERROR, THE TASK COULD NOT BE LEARNED BY THE DETECTOR')
                    break
                else:
                    print('Task learned without issues.')

                # Save the trained model
                save_model(task_detector_i, task_id, 'Models_Task_Recognition/'+str(experiment['experiment_name']))
                print('Training now on the hnet')
                train_losses_, val_losses_, best_w_ =train_current_task(
                                                                    model, 
                                                                    hnet,
                                                                    y_train, 
                                                                    x_train, 
                                                                    y_val,
                                                                    x_val, 
                                                                    calc_reg = calc_reg,
                                                                    cond_id = int(task_id),
                                                                    lr=0.001,
                                                                    lr_step_size=5,
                                                                    lr_gamma= lr_gamma, #0.9
                                                                    sequence_length_LSTM = seq_length_LSTM, #15
                                                                    batch_size_train = batch_size_train, #15
                                                                    batch_size_val = batch_size_train, #15
                                                                    num_epochs=1000, 
                                                                    delta = delta,
                                                                    beta = beta_hnet_reg,             
                                                                    regularizer=reg_hnet,
                                                                    l1_ratio = l1_ratio_reg, #0.5
                                                                    alpha = alpha_reg,    
                                                                    early_stop = 5,
                                                                    chunks = False)
                W_best = hnet(cond_id = task_id)
                r2 = calc_explained_variance_mnet(x_val, y_val, W_best, model)
                print('R2 for the HNET on Task ', task_id, ' is ', r2)
                # Save the trained model
                save_model(hnet, task_id, "HNET_Task_Recog/"+str(experiment['experiment_name']))
        
        print(predicted_tasks)


def main(args):

    index = args.index

    # Load the list of experiments from JSON
    with open(os.path.join('config.json'), 'r') as f:
        experiments = json.load(f)

    experiment = experiments[index]
    name = experiment['experiment_name']
    print('Running esperiment ', name)

    # Loading data
    data = experiment['data']
    data_dir = "./Data/"
    with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
        datasets = pickle.load(fp)

    results_dict = run_experiment(experiment, datasets)

    path_to_results = os.path.join('.','Results')

    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)

    file_path = os.path.join(path_to_results, name'.pkl')
    
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

    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=1,
    #     help="Set the initialization seed",
    # )

    args = parser.parse_args()
    main(args)

