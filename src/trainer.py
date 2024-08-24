import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import *
from copy import deepcopy
import torch.utils.data as data
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import math

from src.helpers import * 
from src.sequence_datasets import *

device = torch.device('cuda:0') #suposed to be cuda
#device = torch.device('cpu') #suposed to be cuda
dtype = torch.float32

    
def train_model(model, X,Y,
                X_val, 
                Y_val,
                lr=0.0001,
                lr_step_size=10,
                lr_gamma=0.9,
                sequence_length_LSTM=10,
                batch_size_train = 3,
                batch_size_val = 3,
                num_epochs=1000, 
                delta = 8,                 
                regularizer=None,
                l1_ratio = 0.5,
                alpha = 1e-5,     
                early_stop = 5,
                
                ):

    # Set up the optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    train_dataset = SequenceDataset(
    Y,    X,    sequence_length=sequence_length_LSTM)

    val_dataset = SequenceDataset(
    Y_val,    X_val,    sequence_length=sequence_length_LSTM)
    loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

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
                X_ = X_.to(device)
                y_ = y_.to(device)
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()

                        output_t = model(X_)
                        output_t = torch.squeeze(output_t)

                        loss_t = huber_loss(output_t, y_, delta = delta)
                        
                        # Add regularization to the loss in the training phase
                        if regularizer is not None:
                            loss_t_r = loss_t + regularizer(model, l1_ratio, alpha)
                        
                        else:
                            loss_t_r = loss_t
                        # Compute gradients and perform an optimization step
                        loss_t_r.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        # just compute the loss in validation phase
                        output_t = model(X_)
                        output_t = torch.squeeze(output_t)

                        loss_t = huber_loss(output_t, y_, delta = delta)

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
                    best_epoch = epoch
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
                            print(best_epoch)
                            return np.array(train_losses), np.array(val_losses)

        # Update learning rate with the scheduler
        scheduler.step()
        print("Epoch {:03} Train {:.4f} Val {:.4f}".format(epoch, train_losses[-1], val_losses[-1]))

    # load best model weights
    model.load_state_dict(best_model_wts)

    print(best_epoch)

    return np.array(train_losses), np.array(val_losses)




def train_hypernet(model, hnet,y_train_base, x_train_base,
                y_train_stim,  x_train_stim,
                y_val_base,  x_val_base,
                y_val_stim,    x_val_stim,
                lr=0.0001,
                lr_step_size=10,
                lr_gamma=0.9,
                sequence_length_LSTM=10,
                batch_size_train = 3,
                batch_size_val = 3,
                num_epochs=1000, 
                delta = 8,                 
                regularizer=None,
                l1_ratio = 0.5,
                alpha = 1e-5,     
                early_stop = 5,
                chunks = False
                
                ):

    
    # Initialize the hypernetwork

    # --> this was only when using th models from hypnettorch
    if chunks:
         hnet.apply_chunked_hyperfan_init(mnet = model)
    else: 
         hnet.apply_hyperfan_init(mnet=model)

    # Set up the optimizer with the specified learning rate
    optimizer = torch.optim.Adam(hnet.internal_params, lr=lr)

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
    train_dataset_baseline = SequenceDataset(
    y_train_base,    x_train_base,    sequence_length=sequence_length_LSTM)

    train_dataset_stim = SequenceDataset(
    y_train_stim,    x_train_stim,    sequence_length=sequence_length_LSTM)

    val_dataset_baseline = SequenceDataset(
    y_val_base,    x_val_base,    sequence_length=sequence_length_LSTM)

    val_dataset_stim = SequenceDataset(
    y_val_stim,    x_val_stim,    sequence_length=sequence_length_LSTM)

    loader_train_b = data.DataLoader(train_dataset_baseline, batch_size=batch_size_train, shuffle=True)
    loader_train_s = data.DataLoader(train_dataset_stim, batch_size=batch_size_train, shuffle=True)

    loader_val_b = data.DataLoader(val_dataset_baseline, batch_size=batch_size_val, shuffle=True)
    loader_val_s = data.DataLoader(val_dataset_stim, batch_size=batch_size_val, shuffle=True)

    # Loop through epochs
    for epoch in np.arange(num_epochs):
        for phase in ['train', 'val']:
            # set model to train/validation as appropriate
            if phase == 'train':
                model.train()
                loaders = zip(loader_train_b, loader_train_s)
            else:
                model.eval()
                loaders = zip(loader_val_b, loader_val_s)

            # Initialize variables to track loss and batch size
            running_loss = 0
            running_size = 0        

            # Iterate over batches in the loader
            for data_b, data_s in loaders:

                # Define data for this batch
                x_b = data_b[0].to('cuda')
                y_b = data_b[1].to('cuda')
                x_s = data_s[0].to('cuda')
                y_s = data_s[1].to('cuda')
               
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()

                        # Compute BASELINE loss.
                        W_base = hnet(cond_id=0)
                        base_P = model.forward(x_b, weights=W_base)
                        base_P = torch.squeeze(base_P) # torch.sigmoid(base_P))
                        loss_base = huber_loss(base_P, y_b, delta = delta)
                        
                        
                        # Compute STIMULATION loss.
                        W_stim = hnet(cond_id=1)
                        stim_P = model.forward(x_s, weights=W_stim)
                        stim_P = torch.squeeze(stim_P) #torch.sigmoid(stim_P))
                        loss_stim = huber_loss(stim_P, y_s, delta = delta)
                        
                        # Combine loss for 2 tasks
                        loss_t = loss_base + loss_stim    #only for printing

                        # Add regularization to the loss in the training phase
                        if regularizer is not None:
                            loss_stim_reg = loss_stim + regularizer(W_stim, l1_ratio, alpha)
                            loss_base_reg = loss_base + regularizer(W_base, l1_ratio, alpha)
                            # Combine loss for 2 tasks
                            loss_t_r = loss_base_reg + loss_stim_reg

                        else:               
                            loss_t_r = loss_t 
                        
                        

                        # Compute gradients and perform an optimization step
                        loss_t_r.backward()
                        optimizer.step()


                else:
                    # just compute the loss in validation phase
                    W_base = hnet(cond_id=0)
                    base_P = model.forward(x_b, weights=W_base)
                    base_P = torch.squeeze(base_P) #torch.sigmoid(base_P))
                    loss_base = huber_loss(base_P, y_b, delta = delta)

                    W_stim = hnet(cond_id=1)
                    stim_P = model.forward(x_s, weights=W_stim)
                    stim_P = torch.squeeze(stim_P) #torch.sigmoid(stim_P))
                    loss_stim = huber_loss(stim_P, y_s, delta = delta)

                    loss_t = loss_base + loss_stim

                # Ensure the loss is finite
                assert torch.isfinite(loss_t)
                assert torch.isfinite(loss_t_r)
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
                            return np.array(train_losses), np.array(val_losses), W_base, W_stim

        # Update learning rate with the scheduler
        scheduler.step()
        print("Epoch {:03} Train {:.4f} Val {:.4f}".format(epoch, train_losses[-1], val_losses[-1]))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return np.array(train_losses), np.array(val_losses), W_base, W_stim


def train_model_EWC(model, X,Y,
                X_val, 
                Y_val,
                fisher_matrices, 
                optimal_params,
                lr=0.0001,
                lr_step_size= 10,
                lr_gamma=0.9,
                sequence_length_LSTM= 10,
                batch_size_train = 3,
                batch_size_val = 3,
                num_epochs=1000, 
                delta = 8,                 
                regularizer=None,
                l1_ratio = 0.5,
                alpha = 1e-5,     
                early_stop = 5,
                lambda_ewc = 0.2):

    # Set up the optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    train_dataset = SequenceDataset(
    Y,    X,    sequence_length=sequence_length_LSTM)

    val_dataset = SequenceDataset(
    Y_val,    X_val,    sequence_length=sequence_length_LSTM)
    loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

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
                X_ = X_.to(device)
                y_ = y_.to(device)
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()

                        output_t = model(X_)
                        output_t = torch.squeeze(output_t)


                        loss_t = huber_loss(output_t, y_, delta = delta)

                        ewc_loss_ = ewc_loss(model, fisher_matrices, optimal_params)
                        loss_t += lambda_ewc * ewc_loss_
                        
                        
                        # Add regularization to the loss in the training phase
                        if regularizer is not None:
                            
                             loss_t += regularizer(model, l1_ratio, alpha)
                        # Compute gradients and perform an optimization step
                        loss_t.backward(retain_graph=True)
                        optimizer.step()
                else:
                    # just compute the loss in validation phase
                    output_t = model(X_)
                    output_t = torch.squeeze(output_t)

                    loss_t = huber_loss(output_t, y_, delta = delta)
                    

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
                    best_epoch = epoch
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
                            print(best_epoch)
                            return np.array(train_losses), np.array(val_losses), best_epoch

        # Update learning rate with the scheduler
        scheduler.step()
        print("Epoch {:03} Train {:.4f} Val {:.4f}".format(epoch, train_losses[-1], val_losses[-1]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    print(best_epoch)

    return np.array(train_losses), np.array(val_losses), best_epoch