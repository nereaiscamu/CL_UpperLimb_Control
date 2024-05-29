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

from src.regularizers import *
from src.trainer import *
from src.helpers import * 
from Models.models import *
from src.sequence_datasets import * 


def train_current_task(
        model, 
        hnet,
        y_train, 
        x_train,
        y_val,
        x_val,
        optimizer,
        scheduler,
        calc_reg = False,
        cond_id = 0,
        lr=0.0001,
        lr_step_size=10,
        lr_gamma=0.9,
        sequence_length_LSTM=15,
        batch_size_train = 25,
        batch_size_val = 25,
        num_epochs=1000, 
        delta = 8,      
        beta=0,           
        regularizer=None,
        l1_ratio = 0.5,
        alpha = 1e-5,  
        early_stop = 10,
        LSTM_ = False,
        chunks = False):
    
    # Compute weights that result from hnet from all previous tasks
    if calc_reg == True:
        reg_targets = get_current_targets(cond_id, hnet)
        prev_hnet_theta = None
        prev_task_embs = None
    
    
    
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
    train_dataset = SequenceDataset(y_train, x_train, sequence_length=sequence_length_LSTM)
    val_dataset = SequenceDataset(y_val, x_val, sequence_length=sequence_length_LSTM)
    loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

    # Initialize h0 and c0 outside the model
    if LSTM_ == True:

        h0 = torch.randn(model.num_layers, batch_size_train, model.hidden_size, device=device) * 0.1
        c0 = torch.randn(model.num_layers, batch_size_train, model.hidden_size, device=device) *0.1 # Initialize cell state
        hx = (h0, c0) 
    else:
        hx = torch.randn(model.num_layers, batch_size_train, model.hidden_size, device=device) * 0.1
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
            for data_ in loader:

                # Define data for this batch
                x = data_[0].to('cuda')
                y = data_[1].to('cuda')

                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()

                        # Forward pass through both models
                        W = hnet(cond_id=cond_id)
                        model = RNN_Main_Model(
                            num_features= model.num_features, 
                            hnet_output = W,  
                            hidden_size = model.hidden_size,
                            num_layers= model.num_layers, 
                            out_dims=model.out_features,  
                            dropout= model.dropout_value, 
                            LSTM_ = LSTM_).to(device)
                                    
                        y_pred = model(x, hx)
                        
                        # Compute loss from the current task
                        loss_task = huber_loss(y_pred, y, delta = delta)
                        
                        # Add regularization from the previous tasks
                        if calc_reg:
                            loss_reg = calc_fix_target_reg(
                                hnet,
                                cond_id,
                                targets=reg_targets,
                                mnet=model,
                                prev_theta=prev_hnet_theta,
                                prev_task_embs=prev_task_embs,)

                            loss_t = loss_task + beta * loss_reg 

                        else:
                            loss_t = loss_task 

                        if regularizer is not None:
                            loss_t = loss_t + regularizer(W,alpha,l1_ratio)
                    
                        # Compute gradients and perform an optimization step
                        loss_t.backward()
                        optimizer.step()
                else:
                    # just compute the loss in validation phase
                    # Compute FIRST loss.
                    W = hnet(cond_id=cond_id)
                    model = RNN_Main_Model(num_features= model.num_features, 
                            hnet_output = W,  
                            hidden_size = model.hidden_size,
                            num_layers= model.num_layers, 
                            out_dims=model.out_features,  
                            dropout= model.dropout_value, 
                            LSTM_ = LSTM_).to(device)
                    y_pred = model(x, hx)
                    loss = huber_loss(y_pred, y, delta = delta)

                    loss_t = loss

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
                    best_w = W
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
                            return np.array(train_losses), np.array(val_losses), best_w

        # Update learning rate with the scheduler
        scheduler.step()
        print("Epoch {:03} Train {:.4f} Val {:.4f}".format(epoch, train_losses[-1], val_losses[-1]))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return np.array(train_losses), np.array(val_losses), best_w