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


class ContinualLearningTrainer:
    def __init__(self, model, hnet, n_contexts, device='cuda'):
        self.model = model
        self.hnet = hnet
        self.device = device
        self.n_contexts = n_contexts
        self.new_context = False
        self.context_error_0 = torch.nn.Parameter(torch.zeros((1)), requires_grad=False)
        self.context_error = [self.context_error_0]
        self.confidence_context = [0]
        self.active_context = 0
        self.thresholds_contexts = torch.nn.Parameter(torch.full((60,), 1.2), requires_grad=False)


    def deviate_from_mean(self, modulation, context):
        N = 100
        k = 15

        # Ensure the context index is valid
        if not (0 <= context < len(self.context_error)):
            raise IndexError(f"Context index {context} is out of range.")
        
        context_errors = self.context_error[context]
        
        # Compute minimum loss in the last k values
        min_loss = torch.min(context_errors[-k:-1].min(), modulation)
        
        # Compute the mean of the last N values
        bar = torch.mean(context_errors[-N:-1])

        
        # Ensure the mean is not zero to avoid division by zero
        if bar.item() == 0:
            raise ValueError("Mean value (bar) is zero, cannot divide by zero.")
        
        # Return whether the deviation exceeds the threshold

        return min_loss / bar > 1.4 #1.01


    def train_current_task(
            self,
            y_train, 
            x_train,
            y_val,
            x_val,
            calc_reg=False,
            cond_id=0,
            lr=0.0001,
            lr_step_size=10,
            lr_gamma=0.9,
            sequence_length_LSTM=15,
            batch_size_train=25,
            batch_size_val=25,
            num_epochs=1000, 
            delta=8,      
            beta=0,           
            regularizer=None,
            l1_ratio=0.5,
            alpha=1e-5,  
            early_stop=10,
            LSTM_=False,
            chunks=False):

        optimizer = torch.optim.Adam(self.hnet.internal_params, lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        best_model_wts = None # deepcopy(self.hnet.state_dict()) --> check if that helps.
        best_loss = 1e8

        torch.autograd.set_detect_anomaly(True)

        train_losses = []
        val_losses = []
        change_detect_epoch = []
        prev_context = []
        prev_min_loss = []
        prev_mean_loss = []
        new_context = []
        new_min_loss = []
        new_mean_loss = []

        not_increased = 0
        end_train = 0
        
        train_dataset = SequenceDataset(y_train, x_train, sequence_length=sequence_length_LSTM)
        val_dataset = SequenceDataset(y_val, x_val, sequence_length=sequence_length_LSTM)
        loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

        hx = torch.randn(self.model.num_layers, batch_size_train, self.model.hidden_size, device=self.device) * 0.1

        

        if self.n_contexts == 0:
            self.n_contexts += 1
        
        # if calc_reg and self.active_context > 0:
        #     reg_targets = get_current_targets(self.active_context, self.hnet)
        #     prev_hnet_theta = None
        #     prev_task_embs = None

        prev_hnet = deepcopy(self.hnet)

        for epoch in range(num_epochs):

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.hnet.train()
                    loader = loader_train
                else:
                    self.hnet.eval()
                    loader = loader_val

                running_loss = 0
                running_size = 0        

                for data_ in loader:
                    x = data_[0].to(self.device)
                    y = data_[1].to(self.device)

                    if phase == "train":
                        with torch.set_grad_enabled(True):
                            optimizer.zero_grad()
                            context = self.active_context
                            self.context_error[self.active_context] = torch.cat(
                                [self.context_error[self.active_context], self.context_error_0], dim=0
                            )

                            W = self.hnet(cond_id=context)
                            model = RNN_Main_Model(
                                num_features=self.model.num_features, 
                                hnet_output=W,  
                                hidden_size=self.model.hidden_size,
                                num_layers=self.model.num_layers, 
                                out_dims=self.model.out_features,  
                                dropout=self.model.dropout_value, 
                                LSTM_=LSTM_
                            ).to(self.device)

                            y_pred = model(x, hx)
                            loss_task = F.huber_loss(y_pred, y, delta=delta)

                            if calc_reg and self.active_context>0:
                                reg_targets = get_current_targets_NC(self.active_context, prev_hnet, len(self.context_error))
                                prev_hnet_theta = None
                                prev_task_embs = None
                                loss_reg = calc_fix_target_reg_NC(
                                    self.hnet,
                                    context,
                                    len(self.context_error),
                                    targets=reg_targets,
                                    mnet=self.model,
                                    prev_theta=prev_hnet_theta,
                                    prev_task_embs=prev_task_embs,
                                )
                                loss_t = loss_task + beta * loss_reg
                               
                            else:
                                loss_t = loss_task 

                            if regularizer is not None:
                                loss_t = loss_t + regularizer(W, alpha, l1_ratio)

                            loss_t.backward()
                            optimizer.step()

                            modulation = loss_task.detach()
                            with torch.no_grad():
                                if self.confidence_context[self.active_context] > 0.9 and  self.deviate_from_mean(modulation, self.active_context):
                           
                                    reactivation = False
                                    self.new_context = True   

                                    if epoch == 0:
                                        for c in range(len(self.context_error)):
                                            self.thresholds_contexts[c] += 0.1                                    
                                
                                    for context in range(len(self.context_error)):
                                        W = self.hnet(cond_id=context)
                                        model = RNN_Main_Model(
                                            num_features=self.model.num_features, 
                                            hnet_output=W,  
                                            hidden_size=self.model.hidden_size,
                                            num_layers=self.model.num_layers, 
                                            out_dims=self.model.out_features,  
                                            dropout=self.model.dropout_value, 
                                            LSTM_=LSTM_
                                        ).to(self.device)
                                        y_pred = model(x, hx)
                                        m = F.huber_loss(y_pred, y, delta=delta)
                                        print(m)
                                        thrs_context = self.thresholds_contexts[context]
                                        print(thrs_context * torch.mean(self.context_error[context][-100:-1]))
                                        change_detect_epoch.append(epoch)
                                        prev_context.append(self.active_context)
                                        prev_min_loss.append(torch.min(self.context_error[self.active_context][-15:-1].min(), modulation).detach().cpu().numpy())
                                        prev_mean_loss.append(torch.mean(self.context_error[self.active_context][-1000:-1]).detach().cpu().numpy())
                                        new_context.append(context)
                                        new_min_loss.append(m.detach().cpu().numpy())
                                        new_mean_loss.append(thrs_context * torch.mean(self.context_error[context][-1000:-1]).detach().cpu().numpy())

                                        
                                        
                                        if m < (thrs_context * torch.mean(self.context_error[context][-1000:-1])):
                                            reactivation = True
                                            self.active_context = context
                                            self.thresholds_contexts[context] = 1.2
                                            break

                                    if not reactivation:
                                        self.confidence_context.append(0)
                                        self.active_context = len(self.context_error)
                                        self.n_contexts += 1
                                        self.context_error.append(self.context_error_0)
                                        prev_hnet = deepcopy(self.hnet)

    
                                else:
                                    self.confidence_context[self.active_context] += (1 - self.confidence_context[self.active_context]) * 0.005
                                    self.context_error[self.active_context][-1] = modulation

                    else:
                        W = self.hnet(cond_id=self.active_context)
                        model = RNN_Main_Model(
                            num_features=self.model.num_features, 
                            hnet_output=W,  
                            hidden_size=self.model.hidden_size,
                            num_layers=self.model.num_layers, 
                            out_dims=self.model.out_features,  
                            dropout=self.model.dropout_value, 
                            LSTM_=LSTM_
                        ).to(self.device)
                        y_pred = model(x, hx)
                        loss = F.huber_loss(y_pred, y, delta=delta)
                        loss_t = loss

                    assert torch.isfinite(loss_t)
                    running_loss += loss_t.item()
                    running_size += 1

                running_loss /= running_size
                if phase == "train":
                    train_losses.append(running_loss)
                else:
                    val_losses.append(running_loss)
                    if running_loss < best_loss:
                        best_loss = running_loss
                        best_model_wts = deepcopy(self.hnet.state_dict())
                        final_active_context = self.active_context
                        final_n_contexts = self.n_contexts
                        final_context_error = self.context_error
                        not_increased = 0
                    else:
                        if epoch > 10:
                            not_increased += 1
                            if not_increased == early_stop:
                                for g in optimizer.param_groups:
                                    g['lr'] = g['lr'] / 2
                                not_increased = 0
                                end_train += 1
                            
                            if end_train == 1:
                                self.hnet.load_state_dict(best_model_wts)
                                self.context_error = final_context_error
                                self.active_context = final_active_context
                                self.n_contexts = final_n_contexts
                                print('Final active context :', self.active_context)
                                return self.hnet, np.array(train_losses), np.array(val_losses),\
                                     change_detect_epoch,prev_context, prev_min_loss,\
                                        prev_mean_loss, new_context, \
                                        new_min_loss,new_mean_loss
            print('Num contexts after epoch ', epoch, len(self.context_error))                
            scheduler.step()

        self.hnet.load_state_dict(best_model_wts)
        self.context_error = final_context_error
        self.active_context = final_active_context
        self.n_contexts = final_n_contexts
        print('Final active context :', self.active_context)
        return self.hnet, np.array(train_losses),  np.array(val_losses),change_detect_epoch,\
              prev_context, prev_min_loss,\
                  prev_mean_loss, new_context, \
                  new_min_loss,new_mean_loss