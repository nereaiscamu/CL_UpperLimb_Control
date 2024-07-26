import random
import numpy as np
import os
import sys
import torch

# Generate simulated perturbed neural data

def remove_neurons(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_removed = int((ratio/100)*num_total_neurons)
    idx_removed = random.sample(list(np.arange(0,num_total_neurons)), num_removed)
    for i in idx_removed:
        sim_data[:,i] = 0
    return sim_data


def shuffle_neurons(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_shuffle = int((ratio/100)*num_total_neurons)
    ind_to_permute = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_shuffle)
    ind_to_permute = np.sort(ind_to_permute)
    permuted_indices = np.random.permutation(ind_to_permute)
    for i, new_i in zip(ind_to_permute, permuted_indices):
        sim_data[:,i] = matrix[:,new_i]
    return sim_data


def add_gain(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_change = int((ratio/100)*num_total_neurons)
    gains = np.random.normal(1, 2, size= num_neurons_to_change)
    ind_to_change = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_change)
    for i, gain in zip(ind_to_change, gains):
        sim_data[:,i] = matrix[:,i]*gain
    return abs(sim_data) # --> changed to abs as FR can't be negative. 18/07/2024


def add_offset(matrix, ratio):
    sim_data = matrix.copy()
    num_total_neurons = matrix.shape[1]
    num_neurons_to_change = int((ratio/100)*num_total_neurons)
    offsets = np.random.normal(0, 25, size= num_neurons_to_change)
    ind_to_change = random.sample(list(np.arange(0,num_total_neurons)), num_neurons_to_change)
    for i, offset in zip(ind_to_change, offsets):
        sim_data[:,i] = matrix[:,i] + offset
    return abs(sim_data) # --> changed to abs as FR can't be negative. 18/07/2024


# Manage data, models and folders

def save_model(model, task_id, path):

    # Check if the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Define the file name
    model_file_name = f"Model_Task_{task_id}.pth"  # Use .pth extension for PyTorch models

    # Save the model
    model_path = os.path.join(path, model_file_name)

    # Save the model using torch.save
    torch.save(model, model_path)


# Creating smaller datasets depending on the trial number
def get_reduced_sets(data, num_trials = -1):
    
    if num_trials == -1:
        new_data = data
        
    else:
        new_data = {}

        for s in data.keys():
            x_train, y_train, x_val, y_val, x_test, y_test = data[s]
            trials_train = []
            trials_val = []
            trials_test = []        
            num_test_trials = min(10, num_trials)

            for i in range(num_trials):
                random.seed()
                trials_train.append(random.randint(0,x_train.shape[0]-1))

            for i in range(num_test_trials):
                random.seed()
                trials_val.append(random.randint(0,x_val.shape[0]-1))
                trials_test.append(random.randint(0,x_test.shape[0]-1))

                x_train_reduced = np.array([x_train[i,:,:] for i in trials_train])
                y_train_reduced = np.array([y_train[i,:,:] for i in trials_train])
                x_val_reduced = np.array([x_val[i,:,:] for i in trials_val])
                y_val_reduced = np.array([y_val[i,:,:] for i in trials_val])
                x_test_reduced = np.array([x_test[i,:,:] for i in trials_test])
                y_test_reduced = np.array([y_test[i,:,:] for i in trials_test])
            
            new_data[s] = [x_train_reduced,
                            y_train_reduced,
                            x_val_reduced,
                            y_val_reduced,
                            x_test_reduced,
                            y_test_reduced]
    return new_data

# Ensuring the first learned task is Baseline. 
def ensure_baseline_first(d):
    keys = list(d.keys())
    if keys[0] != 'Data_0_1':
        keys.remove('Data_0_1')
        keys.insert(0, 'Data_0_1')
    updated_dict = {k: d[k] for k in keys}
    return updated_dict

# Shuffle datasets

def shuffle_sets(datasets):
# Shuffle the dictionnary keys to check the importance of the task order.
    keys_list = list(datasets.keys())
    random.seed()
    random.shuffle(keys_list)
    shuffled_sets = {key: datasets[key] for key in keys_list}
    return shuffled_sets


#######################

# Result Analysis helper functions

def true_task_dict(data):
    max_id = 0
    found_ids = []
    true_task_map = {}
    for d in data.keys():
        new_id = d.split('_')[1]
        if new_id not in found_ids:
            found_ids.append(new_id)
            true_task_map[d] = max_id
            max_id += 1
        else:
            idx_id = found_ids.index(new_id)
            true_task_map[d] = idx_id

    return true_task_map


def build_result_df(results, data):
    dataset = []
    r2_test_detector = []
    r2_test_hnet = []
    y_true_detector = []
    y_pred_detector = []
    y_true_hnet = []
    y_pred_hnet = []
    predicted_task = []
    new_task = [] 
    true_task = []
    train_loss = []
    val_loss = []


    true_task_map = true_task_dict(data)

    for set in results.keys():
        dataset.append(set)
        r2_test_detector.append(results[set]['r2_test_detector'])
        r2_test_hnet.append(results[set]['r2_test_hnet'])
        predicted_task.append(results[set]['predicted_task'])
        new_task.append(results[set]['new_task'])
        true_task.append(true_task_map[set])
        if 'y_true_hnet' in results[set].keys():
            y_true_hnet.append([results[set]['y_true_hnet']])
            y_pred_hnet.append([results[set]['y_pred_hnet']])
            
        else:
            y_true_hnet.append([0])
            y_pred_hnet.append([0])
            
        if 'y_true_detector' in results[set].keys():
            y_true_detector.append([results[set]['y_true_detector']])
            y_pred_detector.append([results[set]['y_pred_detector']])
            train_loss.append([results[set]['hnet_train_losses']])
            val_loss.append([results[set]['hnet_val_losses']])
        else:
            y_true_detector.append([0])
            y_pred_detector.append([0])
            train_loss.append([0])
            val_loss.append([0])

    df = pd.DataFrame({'Dataset':dataset,
                   'True_Task': true_task,
                   'Predicted_Task' : predicted_task,
                   'New_Task': new_task, 
                    'Y_t_detector': y_true_detector,
                    'Y_p_detector':y_pred_detector,  
                    'R2_Detector':r2_test_detector,
                    'Y_t_hnet': y_true_hnet,
                    'Y_p_hnet':y_pred_hnet,  
                    'R2_hnet':r2_test_hnet, 
                    'HNET_training_loss': train_loss,
                    'HNET_val_loss': val_loss})
    return df


def build_catas_forg_df(results, data, models, experiment_name):
    model = []
    test_set = []
    r2_list = []
    data_name = []

    df  = build_result_df(results, data)
    
    for i,m in enumerate(models):
        model_i = torch.load(os.path.join(path_to_hnets,experiment_name, m))
        for task,set in zip(df.True_Task, df.Dataset):
            perturbed_task = set.split('_')[1]

            if perturbed_task == '0':
                name = 'Baseline'
            elif perturbed_task == '1':
                name = 'Removed Neurons'    
            elif perturbed_task == '2':
                name = 'Shuffled Neurons'
            elif perturbed_task == '3':
                name = 'Gain' 
            elif perturbed_task == '4':
                name = 'Offset'

            pred_task = df.loc[df.Dataset == set].Predicted_Task.values
                
            if int(pred_task) <= i :
                W = model_i(cond_id = int(pred_task))
                main_net = RNN_Main_Model(num_features= 130, hnet_output = W,  hidden_size = 300,
                                    num_layers= 1,out_dims=2,  
                                    dropout= 0.2,  LSTM_ = False)
                x_train, y_train, x_val, y_val, x_test, y_test = data[set]
                r2, _ = calc_explained_variance_mnet(x_test, y_test, W, main_net)
                model.append(m)
                test_set.append(set)
                r2_list.append(r2)
                data_name.append(name)    

    df_plot = pd.DataFrame({ 'Model':model,
                        'Name' : data_name,
                        'Dataset':test_set,
                        'R2': r2_list})
    return df_plot

import seaborn as sns
import matplotlib.pyplot as plt

def plot_catas_forg(df_plot):
    set_plot_style()
    # Create the bar plot
    plt.figure()

    sns.barplot(data=df_plot, x='Model', y='R2', hue='Name')#, palette=custom_palette,)#, color = 'Name', ci=None)  # ci=None to remove confidence intervals

    plt.title('R2 Value by Model and Dataset during CL with Hypernetworks', fontsize=20)

    plt.xlabel('Model')

    plt.ylabel('R2 Value')

    new_labels = ['Trained Task 1', 'Trained Task 2', 'Trained Task 3', 'Trained Task 4', 'Trained Task 5']  # New labels for x-axis

    plt.xticks(ticks=range(len(new_labels)), labels=new_labels)
    plt.yticks(fontsize = 12)
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 0.75), loc='upper left')
    plt.tight_layout()
    plt.ylim([0.5,0.9])
    plt.show()