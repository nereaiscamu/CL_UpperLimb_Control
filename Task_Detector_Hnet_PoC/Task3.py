### Imports
import pandas as pd
import numpy as np
import pickle
import json
import argparse

# Imports DL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import *
import torch.utils.data as data
from torch.utils.data import Dataset
from hypnettorch.hnets import HyperNetInterface
from hypnettorch.hnets import HMLP
import copy
import time

from helpers_task_detector import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperimentConfig:
    def __init__(self, experiment):
        self.hidden_units = experiment['hidden_units']
        self.num_layers = experiment['num_layers']
        self.dropout = experiment['dropout']
        self.lr_detector = experiment['lr_detector']
        self.lr_step_size = experiment['lr_step_size']
        self.lr_gamma = experiment['lr_gamma']
        self.seq_length_LSTM = experiment['seq_length_LSTM']
        self.batch_size_train = experiment['batch_size_train']
        self.batch_size_val = experiment['batch_size_val']
        self.delta = experiment['delta']
        self.l1_ratio_reg = experiment['l1_ratio_reg']
        self.alpha_reg = experiment['alpha_reg']
        self.lr_hnet = experiment['lr_hnet']
        self.beta_hnet_reg = experiment['beta_hnet_reg']
        self.thrs = experiment['thrs']
        self.hidden_layers_hnet = experiment['hidden_layers_hnet']
        self.emb_size = experiment['embedding_size']
        self.experiment_name = experiment['experiment_name']


class Run_Experiment_Block3:
    def __init__(self, config, device, datasets):
        self.config = config
        self.device = device
        self.datasets = datasets
        self.num_features = datasets[list(datasets.keys())[0]][0].shape[2]
        self.num_dim_output = datasets[list(datasets.keys())[0]][1].shape[2]
        self.num_conditions = 60
        self.size_task_embedding = 24
        self.hnet = self._initialize_hnet()
        self.model = self._initialize_model()
        self.continual_trainer = ContinualLearningTrainer(self.model, self.hnet, self.num_conditions, self.device)
        self.calc_reg = False
    
    def _initialize_hnet(self):
        param_shapes = [p.shape for p in list(self._initialize_task_detector().parameters())]
        hnet = HMLP(param_shapes, uncond_in_size=0,
                    cond_in_size=self.config.emb_size,
                    layers=self.config.hidden_layers_hnet,
                    num_cond_embs=self.num_conditions).to(self.device)
        for param in hnet.parameters():
            param.requires_grad = True
        hnet.apply_hyperfan_init()
        return hnet

    def _initialize_task_detector(self):
        return Causal_Simple_RNN(
            num_features=self.num_features, 
            hidden_units=self.config.hidden_units, 
            num_layers=self.config.num_layers, 
            out_dims=self.num_dim_output,
            dropout=self.config.dropout
        ).to(self.device)

    def _initialize_model(self):
        w_test = self.hnet(cond_id=0)
        model = RNN_Main_Model(
            num_features=self.num_features, 
            hnet_output=w_test,  
            hidden_size=self.config.hidden_units,
            num_layers=self.config.num_layers,
            out_dims=self.num_dim_output,  
            dropout=self.config.dropout,  
            LSTM_=False
        ).to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def evaluate_model(self, x_train, y_train, x_val, y_val, x_test, y_test, model):
        y_hat, y_true, train_score, v_score, test_score = eval_model(
            x_train, y_train,
            x_val, y_val,
            x_test, y_test, 
            model, 
            metric='r2'
        )
        return y_hat, y_true, train_score, v_score, test_score

    def train_hnet(self, x_train, y_train, x_val, y_val, task_id, calc_reg):
        train_losses, val_losses, best_model = self.continual_trainer.train_current_task(
            y_train, 
            x_train, 
            y_val,
            x_val, 
            calc_reg=self.calc_reg,
            cond_id=int(task_id),
            lr=self.config.lr_hnet,
            lr_step_size=5,
            lr_gamma=self.config.lr_gamma,
            sequence_length_LSTM=self.config.seq_length_LSTM,
            batch_size_train=self.config.batch_size_train,
            batch_size_val=self.config.batch_size_train,
            num_epochs=1000, 
            delta=self.config.delta,
            beta=self.config.beta_hnet_reg, 
            regularizer=reg_hnet,
            l1_ratio=self.config.l1_ratio_reg,
            alpha=0.01,
            early_stop=5,
            chunks=False
        )
        return best_model, train_losses, val_losses

    def run(self):
        results_dict = {}
        for s in self.datasets.keys():
            results_dict_subset = {}
            x_train, y_train, x_val, y_val, x_test, y_test = self.datasets[s]

            path_hnet_models = f'./Models/Models_HNET_Block3/{self.config.experiment_name}'
            os.makedirs(path_hnet_models, exist_ok=True)
            start_time = time.time()
            self.hnet, train_losses_, val_losses_, = self.train_hnet(x_train, y_train, x_val, y_val, 
                                                                     self.continual_trainer.active_context, 
                                                                     calc_reg=self.calc_reg)
            print('Active context:' ,self.continual_trainer.active_context)
            print('num contexts:' , self.continual_trainer.n_contexts)
            # Maybe here not pretend all is the same context.
            W_best = self.hnet(cond_id=self.continual_trainer.active_context)
            r2_test, y_pred_test = calc_explained_variance_mnet(x_test, y_test, W_best, self.model)
            results_dict_subset['y_true_hnet'] = y_test
            results_dict_subset['y_pred_hnet'] = y_pred_test
            results_dict_subset['r2_test_hnet'] = r2_test
            save_model(self.hnet, self.continual_trainer.active_context, path_hnet_models)
            results_dict_subset['hnet_train_losses'] = train_losses_
            results_dict_subset['hnet_val_losses'] = val_losses_
            elapsed_time = time.time() - start_time
            results_dict_subset['training_time'] = elapsed_time

            results_dict[s] = results_dict_subset

        return results_dict
         

def main(args):

    index = args.index
    sort = bool(args.sort)

    # Load the list of experiments from JSON
    with open(os.path.join('config.json'), 'r') as f:
        experiments = json.load(f)

    if index == -1:
        for exp in range(125,128): 
            experiment = experiments[exp]
            name = experiment['experiment_name']
            print('Running esperiment ', name)

            # Loading data
            data = experiment['data']
            data_dir = "./Data/"
            with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
                sets = pickle.load(fp)
            print('Data found')

            if sort:
                print('Sorting the data')
                num_trials = experiment['num_trials']
                # Either keep only a number of trials from the dataset or make sure baseline is the first task
                sets = create_sets(sets, num_trials)
                # Save the data to understand which experiment was run
                path_to_save_data = os.path.join(data_dir, data+'_'+str(num_trials)+'trials.pkl')
                # Pickle the data and save it to file
                with open(path_to_save_data, 'wb') as handle:
                    pickle.dump(sets, handle, protocol=4)

                print("Saving data...")
  
            # Now running experiment on the desired trial number
            print('Running experiment...')
            config = ExperimentConfig(experiment)
            runner = Run_Experiment_Block3(config, device, sets)
            results_dict = runner.run()

            path_to_results = os.path.join('.','Results')
            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results)
            file_path = os.path.join(path_to_results, name+'.pkl')
            # Save the dictionary to a file usnig pickle
            with open(file_path, 'wb') as fp:
                pickle.dump(results_dict, fp)
    else:
        experiment = experiments[index]
        name = experiment['experiment_name']
        print('Running esperiment ', name)
        # Loading data
        data = experiment['data']
        data_dir = "./Data/"
        with open(os.path.join(data_dir, data+'.pkl'), 'rb') as fp:
            sets = pickle.load(fp)

        if sort:
                print('Sorting the data')
                num_trials = experiment['num_trials']
                # Either keep only a number of trials from the dataset or make sure baseline is the first task
                sets = create_sets(sets, num_trials)
                # Save the data to understand which experiment was run
                path_to_save_data = os.path.join(data_dir, data+'_'+str(num_trials)+'trials.pkl')
                # Pickle the data and save it to file
                with open(path_to_save_data, 'wb') as handle:
                    pickle.dump(sets, handle, protocol=4)
                print("Saving data...")

        # Now running experiment on the desired trial number
        print('Running experiment...')
        config = ExperimentConfig(experiment)
        runner = Run_Experiment_Block3(config, device, sets)
        results_dict = runner.run()

        path_to_results = os.path.join('.','Results')
        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)
        file_path = os.path.join(path_to_results, name+'.pkl')
        # Save the dictionary to a file usnig pickle
        with open(file_path, 'wb') as fp:
            pickle.dump(results_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Main script to run experiments" 
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index to iterate over the dictionary",
    )

    parser.add_argument(
        "--sort",
        type=int,
        default=0,
        help="If data needs to be sorted to get the baseline first",
    )

    args = parser.parse_args()
    main(args)

   




