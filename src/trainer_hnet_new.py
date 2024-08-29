import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from copy import deepcopy
from src.sequence_datasets import SequenceDataset
from src.regularizers import get_current_targets, calc_fix_target_reg


class TaskTrainer:
    def __init__(self, model, hnet, train_data, val_data, **kwargs):
        """
        Initializes the TaskTrainer class with necessary parameters.

        Args:
            model (nn.Module): The neural network model to be trained.
            hnet (nn.Module): The hypernetwork generating the weights for the main model.
            train_data (tuple): Tuple containing training targets and inputs.
            val_data (tuple): Tuple containing validation targets and inputs.
            **kwargs: Additional keyword arguments for configuration.
        """
        self.model = model
        self.hnet = hnet
        self.train_data = train_data
        self.val_data = val_data
        
        # Default parameters
        default_params = {
            'calc_reg': False,
            'cond_id': 0,
            'lr': 0.0001,
            'lr_step_size': 10,
            'lr_gamma': 0.9,
            'sequence_length': 15,
            'batch_size_train': 25,
            'batch_size_val': 25,
            'num_epochs': 1000,
            'delta': 8,
            'beta': 0,
            'regularizer': None,
            'l1_ratio': 0.5,
            'alpha': 1e-5,
            'early_stop': 10,
            'lstm': False,
            'chunks': False,
        }

        # Override defaults with any provided kwargs
        default_params.update(kwargs)
        
        # Assign parameters
        self.__dict__.update(default_params)
        
        self.optimizer = torch.optim.Adam(hnet.internal_params, lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_losses = []
        self.val_losses = []
        self.not_increased = 0
        self.end_train = 0
        
        self.best_model_wts = deepcopy(hnet.state_dict())
        self.best_loss = 1e8
        self.best_w = None

        if self.calc_reg:
            self.reg_targets = get_current_targets(self.cond_id, hnet)
            self.prev_hnet_theta = None
            self.prev_task_embs = None

        self.train_loader, self.val_loader = self._prepare_dataloaders()
        self.hx = self._initialize_hidden_state()

    def _prepare_dataloaders(self):
        """
        Prepares the DataLoader objects for training and validation.

        Returns:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        train_dataset = SequenceDataset(self.train_data[0], self.train_data[1], sequence_length=self.sequence_length)
        val_dataset = SequenceDataset(self.val_data[0], self.val_data[1], sequence_length=self.sequence_length)

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=self.batch_size_val, shuffle=True)

        return train_loader, val_loader

    def _initialize_hidden_state(self):
        """
        Initializes the hidden state (and cell state if LSTM is used).

        Returns:
            hx (Tensor or tuple): Initialized hidden state (and cell state if LSTM).
        """
        if self.lstm:
            h0 = torch.randn(self.model.num_layers, self.batch_size_train, self.model.hidden_size, device=self.device) * 0.1
            c0 = torch.randn(self.model.num_layers, self.batch_size_train, self.model.hidden_size, device=self.device) * 0.1
            return (h0, c0)
        else:
            return torch.randn(self.model.num_layers, self.batch_size_train, self.model.hidden_size, device=self.device) * 0.1

    def train(self):
        """
        Executes the training loop for the specified number of epochs.
        
        Returns:
            train_losses (np.array): Array of training losses over epochs.
            val_losses (np.array): Array of validation losses over epochs.
            best_w (Tensor): The best weights obtained during training.
        """
        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    loader = self.train_loader
                else:
                    self.model.eval()
                    loader = self.val_loader

                running_loss = 0.0
                running_size = 0

                for data_ in loader:
                    x = data_[0].to(self.device)
                    y = data_[1].to(self.device)

                    if phase == "train":
                        self.optimizer.zero_grad()
                        W = self.hnet(cond_id=self.cond_id)
                        output = self.model(x, self.hx, weights=W)

                        loss = huber_loss(output, y, delta=self.delta)

                        if self.calc_reg:
                            loss_reg = calc_fix_target_reg(
                                self.hnet,
                                self.cond_id,
                                targets=self.reg_targets,
                                mnet=self.model,
                                prev_theta=self.prev_hnet_theta,
                                prev_task_embs=self.prev_task_embs,
                            )
                            loss += self.beta * loss_reg

                        if self.regularizer:
                            loss += self.regularizer(W, self.alpha, self.l1_ratio)

                        loss.backward()
                        self.optimizer.step()
                    else:
                        with torch.no_grad():
                            W = self.hnet(cond_id=self.cond_id)
                            output = self.model(x, self.hx, weights=W)
                            loss = huber_loss(output, y, delta=self.delta)

                    running_loss += loss.item()
                    running_size += 1

                epoch_loss = running_loss / running_size
                if phase == "train":
                    self.train_losses.append(epoch_loss)
                else:
                    self.val_losses.append(epoch_loss)
                    if epoch_loss < self.best_loss:
                        self.best_loss = epoch_loss
                        self.best_w = W
                        self.best_model_wts = deepcopy(self.hnet.state_dict())
                        self.not_increased = 0
                    else:
                        if epoch > 10:
                            self.not_increased += 1
                            if self.not_increased == self.early_stop:
                                for g in self.optimizer.param_groups:
                                    g['lr'] /= 2
                                self.not_increased = 0
                                self.end_train += 1

                            if self.end_train == 2:
                                self.hnet.load_state_dict(self.best_model_wts)
                                return np.array(self.train_losses), np.array(self.val_losses), self.best_w

            self.scheduler.step()
            print(f"Epoch {epoch+1:03}/{self.num_epochs:03} - Train Loss: {self.train_losses[-1]:.4f} - Val Loss: {self.val_losses[-1]:.4f}")

        self.hnet.load_state_dict(self.best_model_wts)
        return np.array(self.train_losses), np.array(self.val_losses), self.best_w
