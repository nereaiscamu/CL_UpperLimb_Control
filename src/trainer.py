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

device = torch.device('cpu') #suposed to be cuda
dtype = torch.float32

def Regularizer_LSTM(model, alpha=1e-5, l1_ratio=0.5):
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: CausalTemporalLSTM instance
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    w_t = model.lstm.weight_ih_l0
    w_l = model.linear.weight
    w_l_1 = model.linear1.weight

    l1_loss = w_t.abs().sum() + w_l.abs().sum() + w_l_1.abs().sum()
    l2_loss = w_t.pow(2.0).sum() + w_l.pow(2.0).sum() + w_l_1.pow(2.0).sum()

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    return reg.item()


def Regularizer_RNN(model, alpha=1e-5, l1_ratio=0.5):
    """
    Implement an L1-L2 penalty on the norm of the model weights.

    model: CausalTemporalLSTM instance
    alpha: scaling parameter for the regularization.
    l1_ratio: mixing parameter between L1 and L2 loss.

    Returns:
    reg: regularization term
    """
    w_t = model.rnn.weight_ih_l0
    w_l = model.linear.weight
 

    l1_loss = w_t.abs().sum() + w_l.abs().sum() 
    l2_loss = w_t.pow(2.0).sum() + w_l.pow(2.0).sum() 

    reg = l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss

    reg = alpha * reg

    return reg.item()

class SequenceDataset(Dataset):

    def __init__(self, y, X, sequence_length=10):
        """
        Initializes the SequenceDataset.
        
        Args:
            y (torch.Tensor): The target labels for each sequence.
            X (torch.Tensor): The input sequences.
            sequence_length (int): The desired length of each sequence.
        """
        self.sequence_length = sequence_length
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
    
def mean_squared_loss(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between two tensors.
    
    Args:
    - y_true: Tensor containing the true values (ground truth).
    - y_pred: Tensor containing the predicted values.
    
    Returns:
    - mse: Mean Squared Error between y_true and y_pred.
    """
    # Ensure both tensors have the same shape
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"

    # Calculate squared differences between true and predicted values
    squared_errors = (y_true - y_pred)**2

    # Calculate the mean of squared errors
    mse = torch.mean(squared_errors)

    return mse

def huber_loss(X, y, delta=8):
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
        print('Val Score: %.2f RMSE' % (valScore))
        testScore = math.sqrt(mean_squared_error(y_true_test, y_pred_test))
        print('Test Score: %.2f RMSE' % (testScore))

        return y_pred_test, y_true_test,trainScore, valScore, testScore
    
    elif metric == 'ev':
        #Compute explained variance
        ev_train = explained_variance_score(y_true_train, y_pred_train)
        ev_val = explained_variance_score(y_true_val, y_pred_val)
        ev_test = explained_variance_score(y_true_test, y_pred_test)
        print('Train EV: %.2f ' % (ev_train))
        print('Val EV: %.2f ' % (ev_val))
        print('Test EV: %.2f ' % (ev_test))
        return y_pred_test, y_true_test, ev_train, ev_val, ev_test
    

    
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
                #X_ = X_.to('cuda')
                #y_ = y_.to('cuda')
                if phase == "train":
                    with torch.set_grad_enabled(True):
                        optimizer.zero_grad()

                        output_t = model(X_)
                        output_t = torch.squeeze(output_t)


                        loss_t = huber_loss(output_t, y_, delta = delta)
                        
                        # Add regularization to the loss in the training phase
                        if regularizer is not None:
                            loss_t += regularizer(model, l1_ratio, alpha)
                        # Compute gradients and perform an optimization step
                        loss_t.backward()
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
                            return np.array(train_losses), np.array(val_losses)

        # Update learning rate with the scheduler
        scheduler.step()
        print("Epoch {:03} Train {:.4f} Val {:.4f}".format(epoch, train_losses[-1], val_losses[-1]))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return np.array(train_losses), np.array(val_losses)