import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from copy import deepcopy
import torch.utils.data as data
from src.sequence_datasets import SequenceDataset
from Models.models import RNN_Main_Model


class ContinualLearningTrainer:
    def __init__(self, model, hnet, n_contexts=0, device='cuda', cov_window_size=15, max_cov_matrices=20, similarity_threshold=5.0):
        """
        Initializes the ContinualLearningTrainer class.

        Args:
            model (nn.Module): The main model for learning tasks.
            hnet (nn.Module): The hypernetwork generating weights for the main model.
            n_contexts (int): Initial number of contexts.
            device (str): Device to use for training ('cuda' or 'cpu').
            cov_window_size (int): The size of the rolling window for covariance matrices.
            max_cov_matrices (int): The maximum number of covariance matrices to store.
            similarity_threshold (float): The threshold for task similarity detection.
        """
        self.model = model
        self.hnet = hnet
        self.device = device
        self.n_contexts = n_contexts
        self.active_context = 0
        self.new_context = False
        self.thresholds_contexts = torch.nn.Parameter(torch.full((60,), 2.2), requires_grad=False)
        self.context_error = [torch.nn.Parameter(torch.zeros((1)), requires_grad=False)]
        self.confidence_context = [0]

        # Covariance and rolling window settings
        self.task_covariances = []
        self.task_cov_counts = []
        self.rolling_covariances = []
        self.cov_window_size = cov_window_size
        self.max_cov_matrices = max_cov_matrices
        self.similarity_threshold = similarity_threshold

    def train_current_task(self, y_train, x_train, y_val, x_val, **kwargs):
        """
        Train the current task using the hypernetwork and main model.

        Args:
            y_train, x_train: Training data and labels.
            y_val, x_val: Validation data and labels.
            **kwargs: Additional parameters for training.

        Returns:
            Various training statistics and final state.
        """
        # Set default training parameters and override with kwargs
        params = self._get_training_params(**kwargs)

        optimizer, scheduler = self._setup_optimizer_scheduler(params['lr'], params['lr_step_size'], params['lr_gamma'])

        best_model_wts, best_loss = None, 1e8
        train_losses, val_losses = [], []
        context_statistics = self._initialize_context_stats()

        # Initialize hidden states for RNN/LSTM
        hx = self._initialize_hidden_state(params['batch_size_train'], params['lstm'])

        prev_hnet = deepcopy(self.hnet)

        for epoch in range(params['num_epochs']):
            for phase in ['train', 'val']:
                self._train_or_validate_phase(phase, optimizer, params, train_losses, val_losses, context_statistics, hx, prev_hnet)

                # Early stopping logic
                if self._should_early_stop(epoch, params['early_stop'], context_statistics, val_losses[-1], best_loss):
                    best_model_wts, final_state = self._finalize_training(best_model_wts, context_statistics)
                    return final_state

            scheduler.step()

        # Return the final state after all epochs
        best_model_wts, final_state = self._finalize_training(best_model_wts, context_statistics)
        return final_state

    def _get_training_params(self, **kwargs):
        """Set default training parameters and override with any provided in kwargs."""
        params = {
            'calc_reg': False,
            'cond_id': 0,
            'lr': 0.0001,
            'lr_step_size': 10,
            'lr_gamma': 0.9,
            'sequence_length_LSTM': 15,
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
        params.update(kwargs)
        return params

    def _setup_optimizer_scheduler(self, lr, lr_step_size, lr_gamma):
        """Setup optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.hnet.internal_params, lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        return optimizer, scheduler

    def _initialize_hidden_state(self, batch_size, lstm=False):
        """Initialize hidden states for RNN or LSTM."""
        if lstm:
            h0 = torch.randn(self.model.num_layers, batch_size, self.model.hidden_size, device=self.device) * 0.1
            c0 = torch.randn(self.model.num_layers, batch_size, self.model.hidden_size, device=self.device) * 0.1
            return (h0, c0)
        else:
            return torch.randn(self.model.num_layers, batch_size, self.model.hidden_size, device=self.device) * 0.1

    def _initialize_context_stats(self):
        """Initialize tracking variables for context-specific statistics."""
        return {
            'not_increased': 0,
            'end_train': 0,
            'change_detect_epoch': [],
            'prev_context': [],
            'prev_min_loss': [],
            'prev_mean_loss': [],
            'new_context': [],
            'new_min_loss': [],
            'new_mean_loss': [],
            'similarity_scores': [],
        }

    def _train_or_validate_phase(self, phase, optimizer, params, train_losses, val_losses, context_stats, hx, prev_hnet):
        """Handle the training or validation phase."""
        loader_train, loader_val = self._prepare_dataloaders(params['sequence_length_LSTM'], params['batch_size_train'], params['batch_size_val'])
        loader = loader_train if phase == 'train' else loader_val
        self.hnet.train() if phase == 'train' else self.hnet.eval()

        running_loss, running_size = 0, 0

        for data_ in loader:
            x, y = data_[0].to(self.device), data_[1].to(self.device)
            if phase == 'train':
                loss = self._train_batch(x, y, optimizer, hx, params, context_stats, prev_hnet)
            else:
                loss = self._validate_batch(x, y, hx, params)

            running_loss += loss.item()
            running_size += 1

        self._update_losses(phase, running_loss / running_size, train_losses, val_losses)

    def _train_batch(self, x, y, optimizer, hx, params, context_stats, prev_hnet):
        """Train a single batch."""
        optimizer.zero_grad()
        W = self.hnet(cond_id=self.active_context)
        model_output = self._forward_pass(x, hx, W, params['lstm'])

        loss_task = F.huber_loss(model_output, y, delta=params['delta'])
        loss_t = self._apply_regularization_if_needed(loss_task, params, W, prev_hnet)

        loss_t.backward()
        optimizer.step()

        self._update_context(modulation=loss_task.detach(), x=x)
        return loss_t

    def _validate_batch(self, x, y, hx, params):
        """Validate a single batch."""
        W = self.hnet(cond_id=self.active_context)
        model_output = self._forward_pass(x, hx, W, params['lstm'])
        loss = F.huber_loss(model_output, y, delta=params['delta'])
        return loss

    def _forward_pass(self, x, hx, W, lstm):
        """Forward pass through the RNN model."""
        model = RNN_Main_Model(
            num_features=self.model.num_features, 
            hnet_output=W,  
            hidden_size=self.model.hidden_size,
            num_layers=self.model.num_layers, 
            out_dims=self.model.out_features,  
            dropout=self.model.dropout_value, 
            LSTM_=lstm
        ).to(self.device)
        return model(x, hx)

    def _apply_regularization_if_needed(self, loss_task, params, W, prev_hnet):
        """Apply regularization if necessary."""
        if params['calc_reg'] and self.active_context > 0:
            reg_targets = get_current_targets_NC(self.active_context, prev_hnet, len(self.context_error))
            loss_reg = calc_fix_target_reg_NC(
                self.hnet,
                self.active_context,
                len(self.context_error),
                targets=reg_targets,
                mnet=self.model,
                prev_theta=None,
                prev_task_embs=None,
            )
            loss_t = loss_task + params['beta'] * loss_reg
        else:
            loss_t = loss_task

        if params['regularizer']:
            loss_t += params['regularizer'](W, params['alpha'], params['l1_ratio'])
        return loss_t

    def _prepare_dataloaders(self, sequence_length, batch_size_train, batch_size_val):
        """Prepare the DataLoaders for training and validation."""
        train_dataset = SequenceDataset(self.y_train, self.x_train, sequence_length=sequence_length)
        val_dataset = SequenceDataset(self.y_val, self.x_val, sequence_length=sequence_length)
        loader_train = data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        loader_val = data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)
        return loader_train, loader_val

    def _update_losses(self, phase, loss, train_losses, val_losses):
        """Update the loss tracking lists."""
        if phase == 'train':
            train_losses.append(loss)
        else:
            val_losses.append(loss)

    def _update_context(self, modulation, x):
        """Update the context-related variables and handle task detection."""
        self.update_rolling_covariance(x.detach())
        rolling_mean_covariance = self.compute_mean_covariance(self.rolling_covariances)

        if self.confidence_context[self.active_context] > 0.9 and self.deviate_from_mean(modulation, self.active_context):
            self.new_context = True
            reactivation = False

            similar_task_found, similar_task_index = self.is_similar_to_previous_tasks(rolling_mean_covariance)
            if similar_task_found:
                self.active_context = similar_task_index
                reactivation = True

            if not reactivation:
                self._create_new_context()

            self._adjust_thresholds(rolling_mean_covariance, modulation)
        else:
            self.confidence_context[self.active_context] += (1 - self.confidence_context[self.active_context]) * 0.005
            self.context_error[self.active_context][-1] = modulation
            self.update_covariance_for_context(self.active_context)

    def _create_new_context(self):
        """Create a new context when a task change is detected."""
        self.confidence_context.append(0)
        self.active_context = len(self.context_error)
        self.n_contexts += 1
        self.context_error.append(torch.nn.Parameter(torch.zeros((1)), requires_grad=False))
        self.rolling_covariances = []

    def _adjust_thresholds(self, rolling_mean_covariance, modulation):
        """Adjust thresholds and context handling for the detected changes."""
        for context in range(len(self.context_error)):
            W = self.hnet(cond_id=context)
            model_output = self._forward_pass(self.x_train, self.hx, W, params['lstm'])
            loss = F.huber_loss(model_output, self.y_train, delta=params['delta'])

            if not self.is_similar_to_previous_tasks(rolling_mean_covariance)[0] and loss < (self.thresholds_contexts[context] * torch.mean(self.context_error[context][-1000:-1])):
                self.active_context = context
                break

    def _should_early_stop(self, epoch, early_stop, context_stats, val_loss, best_loss):
        """Check whether the training should stop early."""
        if val_loss < best_loss:
            best_loss = val_loss
            context_stats['not_increased'] = 0
        else:
            if epoch > 10:
                context_stats['not_increased'] += 1
                if context_stats['not_increased'] == early_stop:
                    for g in self.optimizer.param_groups:
                        g['lr'] /= 2
                    context_stats['not_increased'] = 0
                    context_stats['end_train'] += 1

        if context_stats['end_train'] == 1:
            return True
        return False

    def _finalize_training(self, best_model_wts, context_stats):
        """Finalize the training, returning the best weights and statistics."""
        self.hnet.load_state_dict(best_model_wts)
        return best_model_wts, (self.hnet, np.array(self.train_losses), np.array(self.val_losses), context_stats)
