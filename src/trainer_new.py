import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np

from src.helpers import *
from src.sequence_datasets import *

device = torch.device('cuda:0')
dtype = torch.float32


class ModelTrainer:
    def __init__(self, model, train_data, val_data, **kwargs):
        """
        Initialize the ModelTrainer with model, training, and validation data.

        Args:
            model (nn.Module): The model to be trained.
            train_data (tuple): Tuple of (X_train, Y_train).
            val_data (tuple): Tuple of (X_val, Y_val).
            kwargs (dict): Additional arguments for training configuration.
        """
        self.model = model.to(device)
        self.X_train, self.Y_train = train_data
        self.X_val, self.Y_val = val_data

        # Training configuration
        self.lr = kwargs.get('lr', 0.0001)
        self.lr_step_size = kwargs.get('lr_step_size', 10)
        self.lr_gamma = kwargs.get('lr_gamma', 0.9)
        self.sequence_length = kwargs.get('sequence_length_LSTM', 10)
        self.batch_size_train = kwargs.get('batch_size_train', 3)
        self.batch_size_val = kwargs.get('batch_size_val', 3)
        self.num_epochs = kwargs.get('num_epochs', 1000)
        self.delta = kwargs.get('delta', 8)
        self.early_stop = kwargs.get('early_stop', 5)
        self.regularizer = kwargs.get('regularizer', None)
        self.l1_ratio = kwargs.get('l1_ratio', 0.5)
        self.alpha = kwargs.get('alpha', 1e-5)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        # DataLoaders
        self.loader_train = DataLoader(SequenceDataset(self.Y_train, self.X_train, sequence_length=self.sequence_length), 
                                       batch_size=self.batch_size_train, shuffle=True)
        self.loader_val = DataLoader(SequenceDataset(self.Y_val, self.X_val, sequence_length=self.sequence_length), 
                                     batch_size=self.batch_size_val, shuffle=True)

    def train_model(self):
        """
        Train the model with the given configuration.
        
        Returns:
            train_losses (np.array): Array of training losses per epoch.
            val_losses (np.array): Array of validation losses per epoch.
            best_epoch (int): The epoch with the best validation loss.
        """
        best_model_wts = deepcopy(self.model.state_dict())
        best_loss = 1e8
        train_losses = []
        val_losses = []
        not_increased = 0
        end_train = 0

        for epoch in np.arange(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loader = self.loader_train
                else:
                    self.model.eval()
                    loader = self.loader_val

                running_loss, running_size = self._run_epoch(loader, phase)

                running_loss /= running_size
                if phase == 'train':
                    train_losses.append(running_loss)
                else:
                    val_losses.append(running_loss)
                    best_loss, not_increased, end_train, best_model_wts = self._check_early_stopping(
                        running_loss, best_loss, epoch, not_increased, end_train, best_model_wts)

                    if end_train == 2:
                        self.model.load_state_dict(best_model_wts)
                        return np.array(train_losses), np.array(val_losses), best_epoch

            self.scheduler.step()
            print(f"Epoch {epoch:03} Train {train_losses[-1]:.4f} Val {val_losses[-1]:.4f}")

        self.model.load_state_dict(best_model_wts)
        return np.array(train_losses), np.array(val_losses), best_epoch

    def _run_epoch(self, loader, phase):
        running_loss = 0
        running_size = 0

        for X_, y_ in loader:
            X_, y_ = X_.to(device), y_.to(device)
            if phase == "train":
                self.optimizer.zero_grad()
                loss = self._compute_loss(X_, y_)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss = self._compute_loss(X_, y_)

            assert torch.isfinite(loss)
            running_loss += loss.item()
            running_size += 1

        return running_loss, running_size

    def _compute_loss(self, X_, y_):
        output = self.model(X_)
        output = torch.squeeze(output)
        loss = huber_loss(output, y_, delta=self.delta)
        
        if self.regularizer:
            loss += self.regularizer(self.model, self.l1_ratio, self.alpha)

        return loss

    def _check_early_stopping(self, running_loss, best_loss, epoch, not_increased, end_train, best_model_wts):
        if running_loss < best_loss:
            best_loss = running_loss
            best_epoch = epoch
            best_model_wts = deepcopy(self.model.state_dict())
            not_increased = 0
        else:
            if epoch > 10:
                not_increased += 1
                if not_increased == self.early_stop:
                    print('Decrease LR')
                    for g in self.optimizer.param_groups:
                        g['lr'] /= 2
                    not_increased = 0
                    end_train += 1
        return best_loss, not_increased, end_train, best_model_wts


class HypernetTrainer(ModelTrainer):
    def __init__(self, model, hnet, train_data, val_data, **kwargs):
        """
        Initialize the HypernetTrainer with model, hypernetwork, training, and validation data.

        Args:
            model (nn.Module): The main model to be trained.
            hnet (nn.Module): The hypernetwork.
            train_data (tuple): Tuple of (X_train, Y_train) for baseline and stim data.
            val_data (tuple): Tuple of (X_val, Y_val) for baseline and stim data.
            kwargs (dict): Additional arguments for training configuration.
        """
        super().__init__(model, train_data, val_data, **kwargs)
        self.hnet = hnet.to(device)
        self.chunks = kwargs.get('chunks', False)

        if self.chunks:
            self.hnet.apply_chunked_hyperfan_init(mnet=self.model)
        else:
            self.hnet.apply_hyperfan_init(mnet=self.model)

    def train_hypernet(self):
        best_loss = 1e8
        train_losses = []
        val_losses = []
        not_increased = 0
        end_train = 0

        for epoch in np.arange(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loaders = zip(self.loader_train_baseline, self.loader_train_stim)
                else:
                    self.model.eval()
                    loaders = zip(self.loader_val_baseline, self.loader_val_stim)

                running_loss, running_size = self._run_epoch_hypernet(loaders, phase)

                running_loss /= running_size
                if phase == 'train':
                    train_losses.append(running_loss)
                else:
                    val_losses.append(running_loss)
                    best_loss, not_increased, end_train, best_model_wts = self._check_early_stopping(
                        running_loss, best_loss, epoch, not_increased, end_train, best_model_wts)

                    if end_train == 2:
                        self.model.load_state_dict(best_model_wts)
                        return np.array(train_losses), np.array(val_losses), self.hnet(cond_id=0), self.hnet(cond_id=1)

            self.scheduler.step()
            print(f"Epoch {epoch:03} Train {train_losses[-1]:.4f} Val {val_losses[-1]:.4f}")

        self.model.load_state_dict(best_model_wts)
        return np.array(train_losses), np.array(val_losses), self.hnet(cond_id=0), self.hnet(cond_id=1)

    def _run_epoch_hypernet(self, loaders, phase):
        running_loss = 0
        running_size = 0

        for (x_b, y_b), (x_s, y_s) in loaders:
            x_b, y_b = x_b.to(device), y_b.to(device)
            x_s, y_s = x_s.to(device), y_s.to(device)

            if phase == "train":
                self.optimizer.zero_grad()
                loss = self._compute_hypernet_loss(x_b, y_b, x_s, y_s)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss = self._compute_hypernet_loss(x_b, y_b, x_s, y_s)

            assert torch.isfinite(loss)
            running_loss += loss.item()
            running_size += 1

        return running_loss, running_size

    def _compute_hypernet_loss(self, x_b, y_b, x_s, y_s):
        W_base = self.hnet(cond_id=0)
        base_P = self.model(x_b, weights=W_base).squeeze()
        loss_base = huber_loss(base_P, y_b, delta=self.delta)

        W_stim = self.hnet(cond_id=1)
        stim_P = self.model(x_s, weights=W_stim).squeeze()
        loss_stim = huber_loss(stim_P, y_s, delta=self.delta)

        total_loss = loss_base + loss_stim
        if self.regularizer:
            total_loss += self.regularizer(W_stim, self.l1_ratio, self.alpha)
            total_loss += self.regularizer(W_base, self.l1_ratio, self.alpha)

        return total_loss


class EWCTrainer(ModelTrainer):
    def __init__(self, model, train_data, val_data, fisher_matrices, optimal_params, **kwargs):
        """
        Initialize the EWCTrainer with model, training, and validation data.

        Args:
            model (nn.Module): The model to be trained.
            train_data (tuple): Tuple of (X_train, Y_train).
            val_data (tuple): Tuple of (X_val, Y_val).
            fisher_matrices (dict): Fisher matrices for EWC.
            optimal_params (dict): Optimal parameters for EWC.
            kwargs (dict): Additional arguments for training configuration.
        """
        super().__init__(model, train_data, val_data, **kwargs)
        self.fisher_matrices = fisher_matrices
        self.optimal_params = optimal_params
        self.lambda_ewc = kwargs.get('lambda_ewc', 0.2)

    def train_model_ewc(self):
        best_loss = 1e8
        train_losses = []
        val_losses = []
        not_increased = 0
        end_train = 0

        for epoch in np.arange(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loader = self.loader_train
                else:
                    self.model.eval()
                    loader = self.loader_val

                running_loss, running_size = self._run_epoch_ewc(loader, phase)

                running_loss /= running_size
                if phase == 'train':
                    train_losses.append(running_loss)
                else:
                    val_losses.append(running_loss)
                    best_loss, not_increased, end_train, best_model_wts = self._check_early_stopping(
                        running_loss, best_loss, epoch, not_increased, end_train, best_model_wts)

                    if end_train == 2:
                        self.model.load_state_dict(best_model_wts)
                        return np.array(train_losses), np.array(val_losses), best_epoch

            self.scheduler.step()
            print(f"Epoch {epoch:03} Train {train_losses[-1]:.4f} Val {val_losses[-1]:.4f}")

        self.model.load_state_dict(best_model_wts)
        return np.array(train_losses), np.array(val_losses), best_epoch

    def _run_epoch_ewc(self, loader, phase):
        running_loss = 0
        running_size = 0

        for X_, y_ in loader:
            X_, y_ = X_.to(device), y_.to(device)
            if phase == "train":
                self.optimizer.zero_grad()
                loss = self._compute_ewc_loss(X_, y_)
                loss.backward(retain_graph=True)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss = self._compute_ewc_loss(X_, y_)

            assert torch.isfinite(loss)
            running_loss += loss.item()
            running_size += 1

        return running_loss, running_size

    def _compute_ewc_loss(self, X_, y_):
        output = self.model(X_).squeeze()
        loss = huber_loss(output, y_, delta=self.delta)

        ewc_loss_ = ewc_loss(self.model, self.fisher_matrices, self.optimal_params)
        loss += self.lambda_ewc * ewc_loss_

        if self.regularizer:
            loss += self.regularizer(self.model, self.l1_ratio, self.alpha)

        return loss

