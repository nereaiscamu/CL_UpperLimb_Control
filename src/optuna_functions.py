import numpy as np

from sklearn.metrics import *
from copy import deepcopy
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.data import Dataset

import optuna


device = torch.device('cpu') #suposed to be cuda
dtype = torch.float32



class CausalTemporalLSTM_Optuna(nn.Module):
    def __init__(self, trial, num_features=124, 
                    out_dims = 6):
        super(CausalTemporalLSTM_Optuna, self).__init__()
        self.num_features = num_features
        self.hidden_units = trial.suggest_int("n_hidden_units", 5,60)
        self.num_layers = trial.suggest_int("num_layers", 1, 5)
        self.input_size = trial.suggest_int('input_size_LSTM', 20, 60)

        self.lstm = nn.LSTM(
            input_size= self.input_size,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers= self.num_layers,
            bidirectional=False,
        )
        self.linear1 = nn.Linear(in_features=self.num_features, out_features=self.input_size)
        self.linear2 = nn.Linear(in_features=self.hidden_units, out_features=out_dims)

        self.dropout1 = nn.Dropout(p= trial.suggest_float('dropout_1', 0, 0.9)) # it was 0.5 for baseline

        self.dropout2 = nn.Dropout(p= trial.suggest_float('dropout_2', 0, 0.9)) 

    def forward(self, x):

        x = self.linear1(x)
        x = self.dropout1(x)
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        output = self.linear2(x)
        
        # Apply sigmoid activation function
        output = torch.sigmoid(output)
        
        return output.squeeze()
    


def Regularizer_LSTM(model, l1_ratio, alpha):
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: CausalTemporalLSTM instance
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    w_t = model.lstm.weight_ih_l0
    w_l_1 = model.linear1.weight
    w_l_2 = model.linear2.weight

    l1_loss = w_t.abs().sum() + w_l_1.abs().sum() + w_l_2.abs().sum()
    l2_loss = w_t.pow(2.0).sum() + w_l_1.pow(2.0).sum() + w_l_2.pow(2.0).sum()

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    return reg.item()



class SequenceDataset(Dataset):

    def __init__(self, y, X, seq_length):
        """
        Initializes the SequenceDataset.
        
        Args:
            y (torch.Tensor): The target labels for each sequence.
            X (torch.Tensor): The input sequences.
            sequence_length (int): The desired length of each sequence.
        """
        self.sequence_length = seq_length
        self.y = torch.tensor(y)
        self.X = torch.tensor(X)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.X.shape[0] * self.X.shape[1]

    def __getitem__(self, i): 
        """
        Gets the i-th sample from the dataset.
        
        Args:
            i (int): Index of the desired sample.
        
        Returns:
            xx (torch.Tensor): Input sequence of length sequence_length.
            yy (torch.Tensor): Corresponding target sequence.
        """
        trial_index = i // self.X.shape[1]
        point_index = i % self.X.shape[1]
        
        if point_index > self.sequence_length - 1:
            point_start = point_index - self.sequence_length
            xx = self.X[trial_index, point_start:point_index, :]
            yy = self.y[trial_index, point_start+1:point_index+1, :]
        else:
            padding_x = self.X[trial_index, 0:1, :].repeat(self.sequence_length - point_index, 1)
            padding_y = self.y[trial_index, 0:1, :].repeat(self.sequence_length - point_index - 1, 1)
            xx = self.X[trial_index, 0:point_index, :]
            xx = torch.cat((padding_x, xx), dim=0)
            yy = self.y[trial_index, 0:point_index + 1, :]
            yy = torch.cat((padding_y, yy), dim=0)
            
        return xx, yy


def huber_loss(X, y, delta):
    
    residual = torch.abs(X - y)
    condition = residual < delta
    loss = torch.where(condition, 0.5 * residual**2, delta * residual - 0.5 * delta**2)
    return loss.mean()



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



def eval_model(xx_train, yy_train, xx_val, yy_val, xx_test, yy_test, model, metric = 'rmse'):

    #Move tensors to cpu and reshape them for evaluation
    y_true_train, y_pred_train = reshape_to_eval(xx_train,yy_train, model)
    y_true_val, y_pred_val = reshape_to_eval(xx_val,yy_val, model)
    y_true_test, y_pred_test = reshape_to_eval(xx_test,yy_test, model)

    if metric == 'rmse':
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_true_train, y_pred_train))
        print('Train Score: %.2f RMSE' % (trainScore))
        valScore = math.sqrt(mean_squared_error(y_true_val, y_pred_val))
        print('Train Score: %.2f RMSE' % (valScore))
        testScore = math.sqrt(mean_squared_error(y_true_test, y_pred_test))
        print('Train Score: %.2f RMSE' % (testScore))

        return y_pred_test, y_true_test,trainScore, valScore, testScore
    
    elif metric == 'ev':
        #Compute explained variance
        ev_train = explained_variance_score(y_true_train, y_pred_train)
        ev_val = explained_variance_score(y_true_val, y_pred_val)
        ev_test = explained_variance_score(y_true_test, y_pred_test)

        return y_pred_test, y_true_test, ev_train, ev_val, ev_test


