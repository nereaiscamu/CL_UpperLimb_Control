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

    def deviate_from_mean(self, modulation, context):
        N = 50
        bar, std = torch.mean(
            self.context_error[context][-N:-1]
        ), torch.std(self.context_error[context][-N:-1])
        return modulation / bar > 1.01

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

        if calc_reg:
            reg_targets = get_current_targets(cond_id, self.hnet)
            prev_hnet_theta = None
            prev_task_embs = None

        best_model_wts = deepcopy(self.model.state_dict())
        best_loss = 1e8

        torch.autograd.set_detect_anomaly(True)

        train_losses = []
        val_losses = []
        not_increased = 0
        end_train = 0
        
        train_dataset = SequenceDataset(y_train, x_train, sequence_length=sequence_length_LSTM)
        val_dataset = SequenceDataset(y_val, x_val, sequence_length=sequence_length_LSTM)
        loader_train = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        loader_val = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

        hx = torch.randn(self.model.num_layers, batch_size_train, self.model.hidden_size, device=self.device) * 0.1

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loader = loader_train
                else:
                    self.model.eval()
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

                            if calc_reg:
                                loss_reg = calc_fix_target_reg(
                                    self.hnet,
                                    context,
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
                            if self.deviate_from_mean(modulation, self.active_context) and self.confidence_context[self.active_context] > 0.9:
                                reactivation = False
                                self.new_context = True
                                for context in range(self.n_contexts):
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
                                    modulation = F.huber_loss(y_pred, y, delta=delta)
                                    if not self.deviate_from_mean(modulation, context):
                                        reactivation = True
                                        self.active_context = context
                                        break

                                if not reactivation:
                                    self.confidence_context.append(0)
                                    self.active_context = len(self.context_error)
                                    self.n_contexts += 1
                                    self.context_error.append(self.context_error_0)
                                    
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
                        best_model_wts = deepcopy(model.state_dict())
                        not_increased = 0
                    else:
                        if epoch > 10:
                            not_increased += 1
                            if not_increased == early_stop:
                                for g in optimizer.param_groups:
                                    g['lr'] = g['lr'] / 2
                                not_increased = 0
                                end_train += 1
                            
                            if end_train == 2:
                                self.model.load_state_dict(best_model_wts)
                                return np.array(train_losses), np.array(val_losses)

            scheduler.step()

        self.model.load_state_dict(best_model_wts)
        return self.model, np.array(train_losses), np.array(val_losses)