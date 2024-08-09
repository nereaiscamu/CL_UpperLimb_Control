import random
import numpy as np
import os
import sys
import torch
import pandas as pd

sys.path.append('../')
from Models.models import *
from src.helpers import *
from src.visualize import *
from src.trainer import *
from src.trainer_hnet import * 


import plotly.express as px
import pickle


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
    #gains = np.random.normal(1, 2, size= num_neurons_to_change)
    gains = np.random.normal(0, 1.25, size= num_neurons_to_change)
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


def create_sets(datasets, num_trials):
    red_sets = get_reduced_sets(datasets, num_trials)
    shuffled_sets = shuffle_sets(red_sets)
    sorted_data = ensure_baseline_first(shuffled_sets)
    return sorted_data


####################################################################

# Result Analysis helper functions





def set_plot_style():
    # Define the custom color palette
    custom_palette = [
        '#5F9EA0', # cadet blue
        '#FFD700', # gold
        '#FFA07A', # light salmon
        '#87CEEB', # light blue
        '#9370DB', # medium purple
        '#98FB98'  # pale green
    ]
    
    # Set the Seaborn palette
    sns.set_palette(custom_palette)
    
    # Set general plot aesthetics
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("whitegrid")

    # Update Matplotlib rcParams for consistent styling
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.title_fontsize': 13,
        'legend.fontsize': 11,
        'axes.titlepad': 20,
        'axes.labelpad': 10,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5
    })


##########################################
############# Result Analysis Task 2


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


def build_result_df(results, data, EWC = False):
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
        if not EWC:
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
            if not EWC:
                y_true_detector.append([results[set]['y_true_detector']])
                y_pred_detector.append([results[set]['y_pred_detector']])
            train_loss.append([results[set]['hnet_train_losses']])
            val_loss.append([results[set]['hnet_val_losses']])
        else:
            y_true_detector.append([0])
            y_pred_detector.append([0])
            train_loss.append([0])
            val_loss.append([0])
    if EWC:
        df = pd.DataFrame({'Dataset':dataset,
                   'True_Task': true_task,
                   'Predicted_Task' : predicted_task,
                   'New_Task': new_task, 
                    'Y_t_hnet': y_true_hnet,
                    'Y_p_hnet':y_pred_hnet,  
                    'R2_hnet':r2_test_hnet, 
                    'HNET_training_loss': train_loss,
                    'HNET_val_loss': val_loss})

    else:

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

def add_name_from_dataset(x):
    perturbed_task = x.split('_')[1]
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
    return name


def build_catas_forg_df(results, data, models, experiment_name, model_type = 'HNET'):
    model = []
    test_set = []
    r2_list = []
    data_name = []

    if model_type == 'HNET':
        path_to_models = './Models/Models_HNET'
        df  = build_result_df(results, data)
    elif model_type == 'EWC':
        path_to_models = './Models/Models_EWC'
        df  = build_result_df(results, data, EWC = True)
    elif model_type == 'FT':
        path_to_models = './Models/Models_FT'
        df  = build_result_df(results, data, EWC = True)
    
    for i,m in enumerate(models):
        model_i = torch.load(os.path.join(path_to_models,experiment_name, m))
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
            x_train, y_train, x_val, y_val, x_test, y_test = data[set]

            if model_type == 'HNET': 
                if int(pred_task) <= i :
                    W = model_i(cond_id = int(pred_task))
                    main_net = RNN_Main_Model(num_features= 130, hnet_output = W,  hidden_size = 300,
                                        num_layers= 1,out_dims=2,  
                                        dropout= 0.2,  LSTM_ = False)
                    
                    r2, _ = calc_explained_variance_mnet(x_test, y_test, W, main_net)
                    model.append(m)
                    test_set.append(set)
                    r2_list.append(r2)
                    data_name.append(name)    
            else:
                if int(pred_task) <= i :
                    y_hat, y_true,train_score, v_score, test_score = eval_model( x_train, y_train,
                                                                        x_val, y_val,
                                                                        x_test, y_test, 
                                                                        model_i, 
                                                                        metric = 'r2')                         
                    model.append(m)
                    test_set.append(set)
                    r2_list.append(test_score)
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
    plt.legend(title='Model',  loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.ylim([0.5,0.9])
    plt.show()

def plot_learning_curves(df_hnet, df_control):

    # Apply the general plot style
    set_plot_style()

    fig, axes = plt.subplots(1, 5, figsize=(16, 6), sharey=True)

    colors = {
        "Training Loss": "#ADD8E6",  # lightblue
        "Validation Loss": "#00008B",  # darkblue
        "Training Loss Control": "#FFA07A",  # lightorange
        "Validation Loss Control": "#FF4500"  # orange-red
    }

    df_hnet_name = df_hnet.copy()
    df_control_name = df_control.copy()
    df_hnet_name['Dataset'] = df_hnet_name['Dataset'].apply(lambda x: add_name_from_dataset(x))
    df_control_name['Dataset'] = df_control_name['Dataset'].apply(lambda x: add_name_from_dataset(x))

    for idx, row in df_hnet_name.iterrows():
        dataset = row.Dataset
        row_control = df_control_name.loc[df_control_name.Dataset == dataset]
        axes[idx].plot(row['HNET_training_loss'][0], label='Training Loss', color=colors['Training Loss'])
        axes[idx].plot(row_control['HNET_training_loss'].values[0][0], label='Training Loss Control', color=colors['Training Loss Control'])
        axes[idx].plot(row['HNET_val_loss'][0], label='Validation Loss', color=colors['Validation Loss'])
        axes[idx].plot(row_control['HNET_val_loss'].values[0][0], label='Validation Loss Control', color=colors['Validation Loss Control'])
        axes[idx].set_title(f'Task {idx + 1} ({row["Dataset"]})')
        axes[idx].set_xlabel('Epoch')
        if idx == 0:
            axes[idx].set_ylabel('Loss')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

def plot_comparison(df):

    set_plot_style()

    # Melting the DataFrame to have a suitable format for plotting
    df_melted = df.melt(id_vars=['Name'], value_vars=['HNET', 'HNET During', 'Single Task Model', 'EWC Model', 'FT Model'],
                        var_name='Model', value_name='R2_Score')
    

    # Custom colors for the models
    custom_colors = [
        '#87CEEB',  # light blue
        '#FFA07A',  # light salmon (orange)
        '#9370DB',  # medium purple
        #'#FF69B4',  # hot pink
        '#FFB6C1',  #Lavender Blush: #FFF0F5
        '#20908d'   # teal   
    ]

    # Setting the plot style
    #sns.set(style="whitegrid")

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        data=df_melted,
        x='Name',
        y='R2_Score',
        hue='Model',
        #palette=custom_colors
    )

    # Adding title and labels
    bar_plot.set_title('Comparison of R² Scores for Different Models per Task')
    bar_plot.set_xlabel('Task Name')
    bar_plot.set_ylabel('R² Score')
   
    plt.legend(title='Model',  loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Setting y-axis limits
    bar_plot.set(ylim=(-0.2, 0.9))

    # Display the plot
    plt.show()

   

def plot_order_heatmap(df):
    # Create a new DataFrame for mean and std
    mean_data = df.pivot_table(index="Second Trained Task", columns="Third Trained Task", values="R2 Third Task", aggfunc='mean')
    std_data = df.pivot_table(index="Second Trained Task", columns="Third Trained Task", values="R2 Third Task", aggfunc='std')

    # Round the mean and std values
    mean_data = mean_data.round(2)
    std_data = std_data.round(5)


    # Prepare annotations with mean and std
    annotations = mean_data.copy().astype(str)  # Convert means to string
    for (i, j), val in np.ndenumerate(std_data):
        annotations.iat[i, j] += f' ± {val:.4f}'  # Append std to the mean

    # Generate the heatmap with enhanced styling
    plt.figure(figsize=(10, 8))

    sns.set(font_scale=1.2)  # Increase the font size
    heatmap = sns.heatmap(mean_data, annot=annotations, cmap="coolwarm", fmt="", linewidths=.5, cbar_kws={'label': 'R2 Value'})

    # Customize the heatmap for better appearance
    plt.title('Heatmap of R2 Values with Std Dev', fontsize=18)
    plt.xlabel('Third Trained Task', fontsize=14)
    plt.ylabel('Second Trained Task', fontsize=14)

    # Improve tick label appearance
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Adjust color bar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    # Show the heatmap
    plt.tight_layout()
    plt.show()

###############3
####### Functions to determine the minimal trial num

def create_task_map(data):
    found_ids = []
    max_id = 0
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


def create_table_results(experiment_name, exp_num_trials):
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

    path_results = './Results/' + experiment_name
    
    with open(os.path.join(path_results +'.pkl'), 'rb') as fp:
        results = pickle.load(fp)
    
    num_trials = exp_num_trials[experiment_name]
    
    if experiment_name == 'Experiment76':
        data_path = './Data/Sim_Data_Experiment60_sorted.pkl'
    elif experiment_name == 'Experiment116':
        data_path = './Data/Sim_Data_Experiment61_-1trials.pkl'
    elif experiment_name == 'Experiment128':
        data_path = './Data/Sim_Data_Experiment61_-1trials.pkl'
    else:
        data_path = './Data/Sim_Data_Experiment61_'+ str(num_trials)+'trials.pkl'
        
    with open(os.path.join(data_path), 'rb') as fp:
        data = pickle.load(fp)
        
    true_task_map = create_task_map(data)
    
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
        else:
            y_true_detector.append([0])
            y_pred_detector.append([0])
        
    df = pd.DataFrame({'Dataset':dataset,
                   'True_Task': true_task,
                   'Predicted_Task' : predicted_task,
                   'New_Task': new_task, 
                    'Y_t_detector': y_true_detector,
                    'Y_p_detector':y_pred_detector,  
                    'R2_Detector':r2_test_detector,
                    'Y_t_hnet': y_true_hnet,
                    'Y_p_hnet':y_pred_hnet,  
                    'R2_hnet':r2_test_hnet}) 
    return df

def average_results(exp_num_trials):
    
    num_trials = []
    acc_detector = []
    r2_detector = []
    for exp in exp_num_trials.keys():
        df = create_table_results(exp, exp_num_trials)
        num_trials.append(exp_num_trials[exp])
        r2_detector.append(df.R2_Detector.mean())
        acc_detector.append((np.sum(df.True_Task == df.Predicted_Task)*10))
    df = pd.DataFrame({'Number Trials':num_trials,
                    'Accuracy Task Detector': acc_detector,
                    'R2 Task Detector' : r2_detector,
                    }) 
    df = df.sort_values(by = 'Number Trials')

    return df

def build_df_exp3(experiment_name, data):

    _,_,_,_,x_test_task2, y_test_task2 = data['Data_4_1']
    _,_,_,_,x_test_task1, y_test_task1 = data['Data_2_2']
    _,_,_,_,x_test_task4, y_test_task4 = data['Data_1_2']
    _,_,_,_,x_test_task3, y_test_task3 = data['Data_3_1']
    _,_,_,_,x_test_task0, y_test_task0 = data['Data_0_1']

    X = [x_test_task0, x_test_task1, x_test_task2, x_test_task3, x_test_task4]
    Y = [y_test_task0, y_test_task1, y_test_task2, y_test_task3, y_test_task4]

    task_ids = []
    random_trials = []
    r2_list = []
    true_ids = []
    num_trials = []

    num_trials_ = [1,2,4,6,8,10,12,14,16,17]
    path_to_detectors ='./Models/Models_Task_Recognition'
    models_exp = np.sort(os.listdir(os.path.join(path_to_detectors, experiment_name)))
    thrs = 0.8

    true_id = 0
    for feats, targs in zip(X,Y):
        for n in num_trials_:
            for t in range(10):
                trials = random.sample(range(17),n)
                x, y = feats[trials, :,:], targs[trials, :,:]
                r2_task = []
                for m in (models_exp):
                    model_i = torch.load(os.path.join(path_to_detectors, experiment_name, m))
                    y_true_test, y_pred_test = reshape_to_eval(x, y, model_i)
                    r2_task.append(r2_score(y_true_test, y_pred_test))
                max_r2 = max(r2_task)
                r2_list.append(max_r2)
                task_id = None
                if max_r2 > thrs:
                    task_id = np.argmax(r2_task)
                task_ids.append(task_id)
                random_trials.append(t)
                num_trials.append(n)
                true_ids.append(true_id)
        true_id += 1
    df = pd.DataFrame({"Trial":random_trials, "Num Trials": num_trials, 
                        "True Task": true_ids, 
                        "Predicted Task" : task_ids,
                        "R2":r2_list })
    return df


def plot_min_trial_leaning(df):
    # Initialize the seaborn style
    sns.set_style("whitegrid")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the accuracy on the primary y-axis
    sns.lineplot(data=df, x='Number Trials', y='Accuracy Task Detector', ax=ax1, marker='o', color='b')
    ax1.set_ylabel('Accuracy', color='b')

    # Create the secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Number Trials', y='R2 Task Detector', ax=ax2, marker='x', color='r')
    ax2.set_ylabel('R2 Score', color='r')

    # Add title
    plt.title('Accuracy and R2 Score vs. Number of Trials')

    # Show plot
    plt.show()


def plot_min_trial_infer(df):
    # Initialize the seaborn style
    sns.set_style("whitegrid")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the accuracy on the primary y-axis
    sns.lineplot(data=df, x='Num Trials', y='Accuracy', ax=ax1, marker='o', color='b')
    ax1.set_ylabel('Accuracy', color='b')

    # Create the secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Num Trials', y='R2', ax=ax2, marker='x', color='r')
    ax2.set_ylabel('R2 Score', color='r')

    # Add title
    plt.title('Accuracy and R2 Score vs. Number of Trials')
    #plt.ylim([0.5,1])

    # Show plot
    plt.show()


#######################
#### Computing forward transfer

def calculate_fwt(accuracies_CL, accuracies_control):
    k = len(accuracies_control)
    fwt_sum = 0.0
    
    for j in range(1, k):  # Start from 1 because we need j=2 to k
        fwt_sum += accuracies_CL[j] - accuracies_control[j]
        
    fwt = fwt_sum / (k - 1)
    return fwt


from hypnettorch.hnets import HMLP

def create_table_FWT(df, experiment_name, data):
    model = []
    test_set = []
    r2_list = []
    data_name = []
    r2_random_mod = []
    path_to_models = './Models/Models_HNET'
    models_exp = np.sort(os.listdir(os.path.join(path_to_models, experiment_name)))
    for i,m in enumerate(models_exp):
        model_i = torch.load(os.path.join(path_to_models,experiment_name, m))
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


            if int(pred_task) == (i+1):
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

                #########################
                ### Now define a random model and compute the R2 before training for each task
                #########################
                
                ####### Define task detector model
                task_detector_model =  Causal_Simple_RNN(num_features=130, 
                                hidden_units= 300, 
                                num_layers = 1, 
                                out_dims = 2,
                                dropout = 0.2).to(device)
        

                #### Defining the template, main and hnet models and initializing them
                
                # We can use the task detector as a template for the main model
                param_shapes = [p.shape for p in list(task_detector_model.parameters())]

                num_conditions = 60 # we want more possible conditions than what we can reach
                size_task_embedding = 8 # seemed to work well 

                hnet = HMLP(param_shapes, uncond_in_size=0,
                    cond_in_size=size_task_embedding,
                    layers=[13], 
                    num_cond_embs=num_conditions).to(device)

                for param in hnet.parameters():
                    param.requires_grad = True

                W_random = hnet(cond_id = int(pred_task))
                r2_random, _ = calc_explained_variance_mnet(x_test, y_test, W_random, main_net)
                r2_random_mod.append(r2_random)

    df_forward_trans = pd.DataFrame({ 'Model':model,
                    'Name' : data_name,
                    'Dataset':test_set,
                    'R2_CL': r2_list,
                    'R2_Random': r2_random_mod})
    
    return df_forward_trans

##########################################
############# Result Analysis Task 3

device = torch.device("cpu")


## 1- Checking Loss Metric

# First using only last model (hnet corresponding to the last embedding), 
# we are going to compute the loss for the whole training dataset,
#  as well as the training loss for the different batches of the testing dataset

def create_dataset_losses(data_experiment, model, keys_training, keys_testing):
    n_contexts = len(keys_training)
    embedding = []
    mean_loss_embeddding = []
    tested_set = []
    loss_set = []
    num_batch = []

    LSTM_ = True

    for i in range(n_contexts):
        x_train, y_train = data_experiment[keys_training[i]][0:2]
        model.to(device)
        W = model(cond_id = int(i))
        main_net = RNN_Main_Model(num_features= 130, hnet_output = W,  
                                hidden_size = 300, num_layers= 1, out_dims=2,  
                                dropout= 0.2,  LSTM_ = LSTM_)
        # Move model to the correct device
        main_net.to(device)
        if LSTM_ == True:
            h0 = torch.randn(1, 50, 300, device= device) * 0.1
            c0 = torch.randn(1, 50, 300, device= device) * 0.1 # Initialize cell state
            hx = (h0, c0) 
        train_dataset = SequenceDataset(y_train, x_train, sequence_length=19)
        loader_train = data.DataLoader(train_dataset, batch_size=50, shuffle=True)
        loss_task = 0
        n_batches = 1
        for x,y in loader_train:
            x = x.to(device)
            y = y.to(device)
            y_pred = main_net(x, hx).to(device)
            loss_task +=  F.huber_loss(y_pred, y, delta=8).detach().numpy()
            n_batches += 1
        loss_real_task = loss_task / n_batches

        for t in range(n_contexts):
            x_train_2, y_train_2 = data_experiment[keys_testing[t]][0:2]
            train_dataset_2 = SequenceDataset(y_train_2, x_train_2, sequence_length=19)
            loader_train_2 = data.DataLoader(train_dataset_2, batch_size=50, shuffle=True)
            n_count = 0
            for x_1, y_2 in loader_train_2:
                x = x_1.to(device)
                y = y_2.to(device)
                y_pred = main_net(x, hx).to(device)
                loss_batch =  F.huber_loss(y_pred, y, delta=8).detach().numpy()
                n_count += 1
                embedding.append('Emb'+ str(i) + '('+(keys_training[i])+')')
                mean_loss_embeddding.append(loss_real_task)
                tested_set.append(keys_testing[t])
                loss_set.append(loss_batch.item())
                num_batch.append(n_count)
                if n_count == 50:
                    break
    df = pd.DataFrame({'Used Embedding' : embedding, 
                    'Mean Task Loss' : mean_loss_embeddding, 
                    'Tested Set' : tested_set, 
                    'Loss Batch' : loss_set,
                    'Num Batch' : num_batch})
    return df


def plot_loss_metric(df):

    # Create the boxplot
    plt.figure(figsize = [12,8])
    set_plot_style()
    ax = sns.boxplot(x='Used Embedding', y='Loss Batch', hue='Tested Set', data=df)

    # Get the unique values and their positions
    unique_embeddings = df['Used Embedding'].unique()
    n_hues = len(df['Tested Set'].unique())
    positions = ax.get_xticks()  # x positions for each category

    # Iterate over each embedding
    for i, embedding in enumerate(unique_embeddings):
        # Calculate the mean task loss for each embedding
        mean_task_loss = df[df['Used Embedding'] == embedding]['Mean Task Loss'].values[0]

        # Define the xmin and xmax for each embedding
        xmin = positions[i] - 0.3
        xmax = positions[i] + 0.1 * (n_hues - 1)

        # Draw the horizontal dashed line
        plt.hlines(y=mean_task_loss, xmin=xmin, xmax=xmax, color='k', linestyle='--', label=f'Mean Task Loss' if i == 0 else "")

    # Add labels and title
    plt.xlabel('Used Embedding')
    plt.ylabel('Loss Batch')
    plt.title('Boxplot of Batch losses vs Mean Loss per Dataset')
    plt.legend(title='Tested Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()




    ## 2- Checking Covariance Metric

def compute_mean_covariance(covariances):
    """Compute the mean of a list of covariance matrices."""
    return sum(covariances) / len(covariances)

def update_covariance_for_context(features, context, rolling_covariances, task_covariances, task_cov_counts):
    """Update the mean covariance matrix for a given context."""
    new_covariance = compute_covariance_matrix(features)

    # Update the rolling window of covariances
    rolling_covariances.append(new_covariance)
    if len(rolling_covariances) > 20: #was 15 before
        rolling_covariances.pop(0)

    # Compute mean of rolling covariances
    rolling_mean_covariance = compute_mean_covariance(rolling_covariances)

    # Update the mean covariance for the context
    if len(task_covariances) <= context:
        task_covariances.append(rolling_mean_covariance)
        task_cov_counts.append(1)
    else:
        task_covariances[context] = update_mean_covariance(
            task_covariances[context],
            rolling_mean_covariance,
            task_cov_counts[context]
        )
        task_cov_counts[context] += 1
    return rolling_covariances, task_covariances, task_cov_counts
    


def create_dataset_cov(data_experiment, keys_training, keys_testing):

    n_contexts = len(keys_training)
    embedding = []
    tested_set = []
    similarity = []
    num_batch = []

    # List to store mean covariance matrices and counts for each context
    task_covariances = []
    task_cov_counts = []

    # Initialize rolling window to store the last 15 covariance matrices
    rolling_covariances = []

    for i in range(n_contexts):
        x_train, y_train = data_experiment[keys_training[i]][0:2]
        train_dataset = SequenceDataset(y_train, x_train, sequence_length=19)
        loader_train = data.DataLoader(train_dataset, batch_size=50, shuffle=True)

        for x,y in loader_train:
            x = x.to(device)
            rolling_covariances, task_covariances, task_cov_counts = update_covariance_for_context(x.detach(), i, rolling_covariances, task_covariances, task_cov_counts)
            
        mean_covariance_task = task_covariances[i]

        # List to store mean covariance matrices and counts for each context
        task_covariances_test = []
        task_cov_counts_test = []

        # Initialize rolling window to store the last 15 covariance matrices
        rolling_covariances_test = []
        
        for t in range(n_contexts):
            x_train_2, y_train_2 = data_experiment[keys_testing[t]][0:2]
            train_dataset_2 = SequenceDataset(y_train_2, x_train_2, sequence_length=19)
            loader_train_2 = data.DataLoader(train_dataset_2, batch_size=50, shuffle=True)
            n_count = 0
            for x_1, y_2 in loader_train_2:
                x = x_1.to(device)          
                rolling_covariances_test, task_covariances_test, task_cov_counts_test = update_covariance_for_context(x, t,rolling_covariances_test, task_covariances_test, task_cov_counts_test )
                n_count += 1
                if n_count>20:
                    rolling_mean_covariance_test = compute_mean_covariance(rolling_covariances_test)
                    diff = torch.abs(rolling_mean_covariance_test - mean_covariance_task)
                    # Calculate similarity score
                    similarity_score = diff.mean().item()
                    embedding.append('Emb'+ str(i) + '('+(keys_training[i])+')')
                    tested_set.append(keys_testing[t])
                    similarity.append(similarity_score)
                    num_batch.append(n_count)
                    if n_count>70:
                        break

    df_covariance = pd.DataFrame({'Used Embedding' : embedding, 
                    'Tested Set' : tested_set, 
                    'Difference Value' : similarity,
                    'Num Batch' : num_batch})
    return df_covariance

def plot_covariance_metric(df):
    # Create the boxplot
    plt.figure(figsize=(12, 8))
    set_plot_style()
    num_embeddings = df['Used Embedding'].nunique()
    ax = sns.boxplot(x='Used Embedding', y='Difference Value', hue='Tested Set', data=df)
    # Draw the horizontal dashed line
    plt.hlines(y=5, xmin=-0.5, xmax=num_embeddings - 0.5, color='r', linestyle='--', label='Similarity Threshold')

    # Get the unique values and their positions
    unique_embeddings = df['Used Embedding'].unique()
    n_hues = len(df['Tested Set'].unique())
    positions = ax.get_xticks()  # x positions for each category

    # Add labels and title
    plt.xlabel('Used Mean Covariance')
    plt.ylabel('Rolling Covariance Batch')
    plt.title('Covariance Matrix differences between datasets')
    plt.legend(title='Tested Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()