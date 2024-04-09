import os
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import *
from copy import deepcopy
import matplotlib.pyplot as plt
import math


from tqdm.auto import tqdm
import seaborn as sns

from src.helpers import *
from src.visualize import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.data import Dataset
import pickle

import optuna
from src.optuna_functions import * 


device = torch.device('cpu') #suposed to be cuda
dtype = torch.float32

import sys
sys.path.append("c:\\Users\\nerea\\OneDrive\\Documentos\\EPFL_MASTER\\PDM\\Project\\PyalData")
# to change for the actual path where PyalData has been cloned


path_to_data = './Data/Processed_Data'
baseline_data = os.path.join(path_to_data, 'Tidy_Sansa_11_04.pkl')

with open(baseline_data, 'rb') as fp:
        df_baseline = pickle.load(fp)


X_train, y_train, X_val, y_val, X_test,\
    y_test, info_train, info_val, info_test = \
    train_test_split(df_baseline, train_variable = 'both_rates', \
                     target_variable = 'target_pos', num_folds = 5)

# Test one of the folds first
fold_num = 'fold0'
fold = 0

print('We are testing the optimization method on fold ', fold)


X_train = X_train[fold_num]
X_val = X_val[fold_num]
X_test = X_test[fold_num]
y_test = y_test[fold_num]
y_train = y_train[fold_num]
y_val = y_val[fold_num]


# Specify that we want our tensors on the GPU and in float32
device = torch.device('cpu') #suposed to be cuda
dtype = torch.float32
path_to_models = './Models'

num_dim_output = y_train.shape[1]
num_features = X_train.shape[1]

seq_length = 75

# Reshape x_train to match the number of columns in the model's input layer
xx_train = X_train.reshape(X_train.shape[0] // seq_length, seq_length, X_train.shape[1])  
# Reshape y_train to match the number of neurons in the model's output layer
yy_train = y_train.reshape(y_train.shape[0] // seq_length, seq_length, y_train.shape[1])  

xx_val = X_val.reshape(X_val.shape[0] // seq_length, seq_length, X_val.shape[1])  
yy_val = y_val.reshape(y_val.shape[0] // seq_length, seq_length, y_val.shape[1])  

xx_test = X_test.reshape(X_test.shape[0] // seq_length, seq_length, X_test.shape[1])  
yy_test = y_test.reshape(y_test.shape[0] // seq_length, seq_length, y_test.shape[1])  

seed = 42
torch.manual_seed(seed)

Reg = globals().get(Regularizer_LSTM)

# Fit the LSTM model
def train_model_optuna(trial):
    
    X = xx_train
    Y = yy_train
    X_val = xx_val
    Y_val = yy_val

    num_epochs= 200
    early_stop = 5

    model = CausalTemporalLSTM_Optuna(trial, num_features= num_features, 
                out_dims = num_dim_output).to(device)
    
    # Set up the optimizer with the specified learning rate
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_name = 'Adam'
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log = True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr)

    lr_step_size = 10 # trial.suggest_int('lr_step_size', 5, 15)
    lr_gamma = trial.suggest_float('lr_gamma', 0.5, 1.3)
    # Set up a learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, 
                                    step_size=lr_step_size, 
                                    gamma=lr_gamma)
        
    # Keep track of the best model's parameters and loss
    best_model_wts = deepcopy(model.state_dict())
    best_loss = 1e8

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Track the train and validation loss
    train_losses = []
    val_losses = []
    # Counters for early stopping
    not_increased = 0
    end_train = 0
    
    # Reshape data for the LSTM
    seq_length_LSTM = trial.suggest_int('seq_length_LSTM', 5, 15)
    train_dataset = SequenceDataset(Y,X,seq_length_LSTM)
    val_dataset = SequenceDataset(Y_val,X_val,seq_length_LSTM)


    batch_size_train = trial.suggest_int('batch_size_train', 25, 75)
    batch_size_val = trial.suggest_int('batch_size_val',25, 75)
    loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

    # hyperparameter for huber loss
    delta = 8 #trial.suggest_int('huber_delta', 5, 10)

    # hyperparameter for regularizer
    alpha = 1e-5 #trial.suggest_float('alpha_reg', 1e-7, 1e-3, log = True)
    l1_ratio = trial.suggest_float('l1_ratio_reg', 0.3, 0.7)

    # Loop through epochs
    for epoch in np.arange(num_epochs):
        for phase in ['train', 'val']:
            # set model to train/validation as appropriate
            if phase == 'train':
                model.train()
                loader = loader_train
            else:
                model.eval()
                loader = loader_val

            # Initialize variables to track loss and batch size
            running_loss = 0
            running_size = 0        

            # Iterate over batches in the loader
            for X_, y_ in loader:
                #X_ = X_.to('cuda')
                #y_ = y_.to('cuda')
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()

                        output_t = model(X_)
                        output_t = torch.squeeze(output_t)


                        loss_t = huber_loss(output_t, y_, delta)
                        
                        # Add regularization to the loss in the training phase
                        loss_t += Regularizer_LSTM(model, l1_ratio, alpha)

                        # Compute gradients and perform an optimization step
                        loss_t.backward()
                        optimizer.step()
                else:
                    # just compute the loss in validation phase
                    output_t = model(X_)
                    output_t = torch.squeeze(output_t)

                    loss_t = huber_loss(output_t, y_, delta)

                # Ensure the loss is finite
                assert torch.isfinite(loss_t)
                running_loss += loss_t.item()
                running_size += 1

            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
            running_loss /= running_size
            if phase == "train":
                train_losses.append(running_loss)
            else:
                val_losses.append(running_loss)
                
                # Update best model parameters if validation loss improves
                if running_loss < best_loss:
                    best_loss = running_loss
                    best_model_wts = deepcopy(model.state_dict())
                    not_increased = 0
                else:
                    # Perform early stopping if validation loss doesn't improve
                    if epoch > 10:
                        not_increased += 1
                        # print('Not increased : {}/5'.format(not_increased))
                        if not_increased == early_stop:
                            print('Decrease LR')
                            for g in optimizer.param_groups:
                                g['lr'] = g['lr'] / 2
                            not_increased = 0
                            end_train += 1
                        
                        if end_train == 2:
                            model.load_state_dict(best_model_wts)
                            y_true_val, y_pred_val = reshape_to_eval(xx_val,yy_val, model)
                            ev_val = explained_variance_score(y_true_val, y_pred_val)
                            return ev_val  # here change to not return a list but a single value for the trial to analyze
                        

        # Update learning rate with the scheduler
        scheduler.step()
        print("Epoch {:03} Train {:.4f} Val {:.4f}".format(epoch, train_losses[-1], val_losses[-1]))


        y_true_val_epoch, y_pred_val_epoch = reshape_to_eval(xx_val,yy_val, model)
        ev_val_epoch = explained_variance_score(y_true_val_epoch, y_pred_val_epoch)
        trial.report(ev_val_epoch, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    y_true_val, y_pred_val = reshape_to_eval(xx_val,yy_val, model)
    ev_val = explained_variance_score(y_true_val, y_pred_val)

    return ev_val



def reshape_to_eval(x,y, model):
    to_t_eval =  lambda array: torch.tensor(array, device='cpu', dtype=dtype)  
    x = to_t_eval(x) 
    y = to_t_eval(y)
    y_pred = model(x)
    y_array = y.detach().cpu().numpy()
    y_pred_array = y_pred.detach().cpu().numpy()

    # Reshape tensors to 2D arrays (flatten the batch and sequence dimensions)
    y_pred_2D = y_pred_array.reshape(-1, y_pred_array.shape[-1])
    y_true_2D = y_array.reshape(-1, y_array.shape[-1])
    
    return y_true_2D, y_pred_2D



if __name__ == "__main__":
        study = optuna.create_study(direction="maximize")
        study.optimize(train_model_optuna, n_trials=100)

        importance_scores = optuna.importance.get_param_importances(study)

        # Print importance scores
        for param, score in importance_scores.items():
            print(f"{param}: {score}")

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]



        print('Study statistics: ')
        print("Number of finished trials: ", len(study.trials))
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))

        print('Best trial: ')
        trial = study.best_trial

        print("Loss: {}".format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))


        ### Plotting the study

        # Plotting the optimization history of the study.

        optuna.visualization.plot_optimization_history(study)

        #Plotting the accuracies for each hyperparameter for each trial.
            
        optuna.visualization.plot_slice(study)

        # Plotting the accuracy surface for the hyperparameters involved in the random forest model.

        optuna.visualization.plot_contour(study, params=["num_layers", "n_hidden_units"]) 