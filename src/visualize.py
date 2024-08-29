import pandas as pd
import numpy as np
import xarray as xr

import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.append("c:\\Users\\nerea\\OneDrive\\Documentos\\EPFL_MASTER\\PDM\\Project\\PyalData")
from pyaldata import *

# ==========================================================
# FUNCTIONS FOR VISUALIZING TRAJECTORIES AND ANGLES
# ==========================================================

def visualize_traj(df, pos_vars, marker_names):
    """ 
    This function is used to visualize the trajectories for given
    markers and given directions, split by reach trial.
    It makes sure that the windows have been accurately created and that 
    they include all the studies trajectories.
    
    Inputs:
        - df: DataFrame containing the trajectory data split by reach trials
        - pos_vars: list of variables to plot, can be ['x', 'y', 'z']
        - markers: list of numbers of the markers to visualize (from 0 to 5)
        
    Returns:
        Subplots of the different trajectories, split by reach trial, 
        with color code for the different trials (each trial includes several reaches)
        A single legend for the entire plot
        Also plots two different mean trajectories for each combination of pos_vars and markers

        
    """
    num_rows = len(pos_vars)
    num_cols = len(marker_names)
    
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

    # Collect handles and labels for the legend
    legend_colors = ['C0', 'C1']  # Define the colors for each legend entry
    legend_labels = ['Trial 0', 'Trial 1']  # Define the labels for each legend entry

    for i, pos_var in enumerate(pos_vars):
        for j, marker in enumerate(marker_names):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(f'{marker}', fontsize = 18)
            if j == 0:
                ax.set_ylabel(f'{pos_var.upper()} Position (mm)', fontsize = 18)
            
            mean_trajectories = []  # Initialize list to store mean trajectories
            
            for idx, t in enumerate(df[pos_var][:2]):
                color = f'C{idx}'  # Use a unique color for each trial
                label = f'{pos_var.upper()} Marker {marker} Trial {idx}'
                mean_trajectory = None  # Initialize mean trajectory
                
                for r in t:
                    for index_d,data in enumerate(r):
                        time = (np.arange(0, len(data[:, j])) * 20)/1000
                        ax.plot(time, data[:, j], color=color, label=label, alpha=0.4)
                        
                        # Compute mean trajectory
                        if mean_trajectory is None:
                            mean_trajectory = np.zeros_like(data[:, j])
                        mean_trajectory += data[:, j]
                
                # Store mean trajectory
                mean_trajectory /= (index_d+1) # Normalize by the number of trials
                
                #print(mean_trajectory[:10])
                mean_trajectories.append(mean_trajectory)
                #print(mean_trajectories)
            
            # Plot two different mean trajectories
            for k, mean_trajectory in enumerate(mean_trajectories):
                ax.plot(time, mean_trajectory, color=f'C{k}', linestyle='--', label=f'Mean Trajectory {k}')
            
            if i == 2:
                ax.set_xlabel('Time (s)', fontsize = 18)
            
    # Create a single legend outside the subplots
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(legend_colors, legend_labels)]
    legend_handles.extend([plt.Line2D([0], [0], color=f'C{k}', linestyle='--', lw=2, label=f'Mean Trajectory {k}') for k in range(len(mean_trajectories))])
    # Create a single legend for the entire figure at the bottom
    fig.legend(legend_handles, legend_labels + [f'Mean Trajectory {k}' for k in range(len(mean_trajectories))],
                loc='lower center', ncol=len(legend_labels)+2, bbox_to_anchor=(0.5, -0.1), fontsize = 'xx-large')
    plt.suptitle('X, Y and Z trajectories for Baseline Trials', fontsize='24')
    plt.tight_layout()

    plt.show()

def visualize_angles(df, var = 'angles', angle_names = ['Shoulder', 'Elbow', 'Wrist']):
    """ 
    This function visualizes the trajectories for given angles split by reach trial.

    Inputs:
        - df: DataFrame containing the trajectory data split by reach trials
        - var: name of the variable to plot, default angles.
        - angle_names: list of angle names (e.g., ['angle_1', 'angle_2', 'angle_3'], to use for the plot titles. Default 'Shoulder', 'Elbow', 'Wrist'


    Returns:
        Subplots of the different angular trajectories, split by reach trial, with color code for the different trials
        A single legend for the entire plot
        Plots the mean trajectory for each combination of angle and trial
    """
    cols = len(angle_names)
    fig, axes = plt.subplots(1, cols, figsize=(20, 5))


    # Collect handles and labels for the legend
    legend_colors = ['C0', 'C1']  # Define the colors for each legend entry
    legend_labels = ['Trial 0', 'Trial 1']  # Define the labels for each legend entry


    for i, angle in enumerate(angle_names):
        ax = axes[i]
        ax.set_title(f'{angle} vs Time')

        if i == 0:
            ax.set_ylabel('Angle (degrees)')

        mean_trajectories = []  # Initialize list to store mean trajectories

        for idx, t in enumerate(df[var][:2]):

            color = f'C{idx}'  # Use a unique color for each trial
            label = f'{var} angle {angle} Trial {idx}'
            mean_trajectory = None  # Initialize mean trajectory
            
            for r in t:
                
                for index_d,data in enumerate(r):
                    time = (np.arange(0, len(data[:, i])) * 20)/1000
                    ax.plot(time, data[:, i], color=color, label=label, alpha=0.4)
                    
                    # Compute mean trajectory
                    if mean_trajectory is None:
                        mean_trajectory = np.zeros_like(data[:, i])
                    mean_trajectory += data[:, i]

            # Store mean trajectory
            mean_trajectory /= (index_d+1) # Normalize by the number of trials
            
            #print(mean_trajectory[:10])
            mean_trajectories.append(mean_trajectory)
            #print(mean_trajectories)

            # Plot two different mean trajectories
            for k, mean_trajectory in enumerate(mean_trajectories):
                ax.plot(time, mean_trajectory, color=f'C{k}', linestyle='--', label=f'Mean Trajectory {k}')
            
            ax.set_xlabel('Time (s)')
            
    # Create a single legend outside the subplots
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(legend_colors, legend_labels)]
    legend_handles.extend([plt.Line2D([0], [0], color=f'C{k}', linestyle='--', lw=2, label=f'Mean Trajectory {k}') for k in range(len(mean_trajectories))])
    # Create a single legend for the entire figure at the bottom
    fig.legend(legend_handles, legend_labels + [f'Mean Trajectory {k}' for k in range(len(mean_trajectories))],
                loc='lower center', ncol=len(legend_labels)+2, bbox_to_anchor=(0.5, -0.1), fontsize = 'xx-large')
    plt.suptitle('Angular  trajectories for baselie trials (no stimulation)', fontsize='24')
    plt.tight_layout()

    plt.show()

# ==========================================================
# FUNCTION TO PLOT TRAINING AND VALIDATION LOSSES
# ==========================================================

def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses over epochs.

    Args:
    - train_losses (numpy array): Array containing training losses for each epoch.
    - val_losses (numpy array): Array containing validation losses for each epoch.
    """
    epochs = len(train_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================================
# FUNCTIONS FOR VISUALIZING MODEL PREDICTIONS
# ==========================================================

def visualize_LSTM(y_hat, y_true, seq_length):

    """
    Visualize LSTM model predictions versus true values for x and y positions.

    Inputs:
        - y_hat: Predicted values from the LSTM model
        - y_true: True values to compare against
        - seq_length: Length of the sequence used in LSTM

    Returns:
        - A set of plots comparing predicted and true values for several trials
    """

    y_hat = y_hat.reshape(y_hat.shape[0] // seq_length, seq_length, y_hat.shape[1])  
    y_true = y_true.reshape(y_true.shape[0] // seq_length, seq_length, y_true.shape[1])  
    
    num_trials = y_hat.shape[0]
    # Define time vector (assuming each sample represents 20ms)
    trial_len = y_true.shape[1]
    time_vector = np.arange(0, trial_len * 0.02, 0.02)  # Time vector in seconds

    # Create a figure and axis objects
    fig, ax = plt.subplots(nrows=2, ncols=5, sharey='row', figsize=[4*5, 10])

    # Plot the signals with vertical spacing
    spacing = 0.5  # Adjust the spacing value as desired

    # Define a list of colors
    colors = ['g', 'b', 'orange']

    # Plot each variable (x, y, z) in separate rows
    for j in range(num_trials)[:5]:
        data = y_true[j]
        pred_data = y_hat[j]
        
        # Plot x variable
        ax[0, j].plot(time_vector, data[:, 0], c=colors[0], label='True')
        ax[0, j].plot(time_vector, pred_data[:, 0], c=colors[0], alpha=0.5, linestyle='--', label='Predicted')
        ax[0, j].set_title('Trial {}'.format(j+1), fontsize='xx-large')
        ax[0, j].spines[['right', 'top', 'left']].set_visible(False)
        
        # Plot y variable
        ax[1, j].plot(time_vector, data[:, 1], c=colors[1])
        ax[1, j].plot(time_vector, pred_data[:, 1], c=colors[1], alpha=0.5, linestyle='--')
        ax[1, j].spines[['right', 'top', 'left']].set_visible(False)
        


    # Set y-label only for the first column
    fig.text(0.075, 0.78, 'X Position', va='center', rotation='vertical', fontsize = 16)
    fig.text(0.075, 0.5, 'Y Position', va='center', rotation='vertical', fontsize = 16)



    # Create a common legend for all subplots
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize='large')
    for line in legend.get_lines():
        line.set_color('k')  # Set legend line color to red

    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()



def visualize_all_models(y_hat1, y_true1, y_hat2,
                        y_hat3, seq_length):
    """
    Visualize model predictions from multiple models (e.g., RNN, LSTM, Linear) versus true values for x and y velocities.

    Inputs:
        - y_hat1: Predicted values from the first model (e.g., RNN)
        - y_true1: True values to compare against
        - y_hat2: Predicted values from the second model (e.g., LSTM)
        - y_hat3: Predicted values from the third model (e.g., Linear)
        - seq_length: Length of the sequence used in the models

    Returns:
        - A set of plots comparing predicted and true values for multiple models across several trials
    """
    y_hat1 = y_hat1.reshape(y_hat1.shape[0] // seq_length, seq_length, y_hat1.shape[1])  
    y_true1 = y_true1.reshape(y_true1.shape[0] // seq_length, seq_length, y_true1.shape[1])  
    y_hat2 = y_hat2.reshape(y_hat2.shape[0] // seq_length, seq_length, y_hat2.shape[1])      
    y_hat3 = y_hat3.reshape(y_hat3.shape[0] // seq_length, seq_length, y_hat3.shape[1])  
    
    num_trials = y_hat1.shape[0]
    # Define time vector (assuming each sample represents 20ms)
    trial_len = y_true1.shape[1]
    time_vector = np.arange(0, trial_len * 0.02, 0.02)  # Time vector in seconds

    # Create a figure and axis objects
    fig, ax = plt.subplots(nrows=2, ncols=5, sharey='row', figsize=[4*5, 10])

    # Plot the signals with vertical spacing
    spacing = 0.5  # Adjust the spacing value as desired

    # Define a list of colors
    colors = ['b', 'green', 'orange']

    # Plot each variable (x, y, z) in separate rows
    for j in range(num_trials)[:5]:
        data1 = y_true1[j]
        pred_data1 = y_hat1[j]

        pred_data2 = y_hat2[j]

        pred_data3 = y_hat3[j]
        
        # Plot x variable
        ax[0, j].plot(time_vector, data1[:, 0], linewidth = 2.5, c='k', label='True')
        ax[0, j].plot(time_vector, pred_data1[:, 0], linewidth = 2.5, c=colors[0], alpha=0.6, linestyle='--', label='Predicted RNN')
    
        ax[0, j].plot(time_vector, pred_data2[:, 0], linewidth = 2.5, c=colors[1], alpha=0.6, linestyle='--', label='Predicted LSTM')

        ax[0, j].plot(time_vector, pred_data3[:, 0], linewidth = 2.5, c=colors[2], alpha=0.6, linestyle='--', label='Predicted Linear')
        ax[0, j].set_title('Trial {}'.format(j+1), fontsize='xx-large')
        ax[0, j].spines[['right', 'top', 'left']].set_visible(False)
        
        # Plot y variable
        ax[1, j].plot(time_vector, data1[:, 1], linewidth = 2.5, c='k', label='True')
        ax[1, j].plot(time_vector, pred_data1[:, 1], linewidth = 2.5, c=colors[0], alpha=0.6, linestyle='--', label='Predicted RNN')
     
        ax[1, j].plot(time_vector, pred_data2[:, 1], linewidth = 2.5, c=colors[1], alpha=0.6, linestyle='--', label='Predicted LSTM')

        ax[1, j].plot(time_vector, pred_data3[:, 1], linewidth = 2.5, c=colors[2], alpha=0.6, linestyle='--', label='Predicted Linear')
        ax[1, j].spines[['right', 'top', 'left']].set_visible(False)

        ax[1, j].set_xlabel("Time (s)", fontsize = 20)  

    # Set y-label only for the first column
    fig.text(0.075, 0.7, 'X Velocity', va='center', rotation='vertical', fontsize = 25)
    fig.text(0.075, 0.3, 'Y Velocity', va='center', rotation='vertical', fontsize = 25)

    # Create a common legend for all subplots
    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize='xx-large')

    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()