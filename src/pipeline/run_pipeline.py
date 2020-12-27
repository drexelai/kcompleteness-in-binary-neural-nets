# Author: Isamu Isozaki, Yigit Alparslan
# Date: 2020/11/10
# Purpose: Train and test for models with hidden layer at n dimensions
from pipeline.get_data import get_titanic_data, get_data_churn_rate
from pipeline.train import train
from pipeline.test import test
from pipeline.argument_parser import parse_arguments
import pandas as pd
import os
import sys
import time
COLUMNS = [
  'train_accuracy', 
  'val_accuracy', 
  'test_accuracy', 
  'area_under_curve', 
  'precision',
  'recall',
  'F1', 
  'sparsity_score',
  'hidden_layers']

# PIPELINE METHODS
# Run pipe and returns train accuracy and test accuracy
def run_pipe(**args):
    prepare_data = get_data_churn_rate if args["use_churn_data"] else get_titanic_data
    X_train, X_test, y_train, y_test = prepare_data(**args)
    model, train_accuracy, val_accuracy, history = train(X_train, y_train, **args)
    test_accuracy, area_under_curve, precision, recall, F1 = test(X_test, y_test, model, **args)
    return train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model, history

def plot_accuracy_graphs_given_history(ax, history, hidden_layer_dimensions):
    # Plot training/validation/testing/ accuracy
    hidden_layer_dimensions = str(hidden_layer_dimensions)
    ax.plot(history.history['accuracy'],label='training accuracy' + hidden_layer_dimensions, color = "blue")
    ax.plot(history.history['val_accuracy'],label='validation accuracy' + hidden_layer_dimensions, color = "red")
    ax.legend(loc=0)
    ax.set_xlabel('epochs')
    ax.set_xlim([0,len(history.history['accuracy'])])
    ax.set_ylabel('Training Accuracies')
    ax.grid(True)
    ax.set_title("Training and Validation Accuracy")
    
def plot_loss_graphs_given_history(ax, history, hidden_layer_dimensions):
    # Plot training/validation/testing/ loss
    hidden_layer_dimensions = str(hidden_layer_dimensions)
    ax.plot(history.history['loss'],label='training loss'+ hidden_layer_dimensions)
    ax.plot(history.history['val_loss'],label='validation loss'+ hidden_layer_dimensions)
    ax.legend(loc=0)
    ax.set_xlabel('epochs')
    ax.set_xlim([0,len(history.history['accuracy'])])
    ax.set_ylabel('Losses')
    ax.grid(True)
    ax.set_title("Training and Validation loss")
    

# Run pipe with parameter hidden_layers and returns train and test accuracy for brute_force
def run_entire_pipeline_for_brute_force_search(**args):
    exp_data = pd.DataFrame(columns = COLUMNS)
    search_space = generate_search_space(args['ihls'], args['df'])
    reduced_search_space = brute_force_search(search_space)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for row in reduced_search_space:
        for model_architecture in row:
            df = model_architecture[0]
            ihls = model_architecture[1]
            args = parse_arguments([''])
            args['hidden_layers'] = calculate_model_architecture(df, ihls)
            train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, _, history = run_pipe(**args)
            # Accumulate plots for each model fits on a figure
            plot_accuracy_graphs_given_history(ax1, history, args['hidden_layers'])
            plot_loss_graphs_given_history(ax2, history, args['hidden_layers'])
            # Calculate sparsity score
            sparsity_score = calculate_sparsity_score(0.5, ihls, 11, df)
            # Append the current model results to csv
            current_row = pd.Series([train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, sparsity_score, args['hidden_layers']], index=COLUMNS)
            exp_data = exp_data.append(current_row, ignore_index=True)
            exp_data.to_csv(os.path.join("..", args['results_dir'], 'brute_force_results.csv'))
    # Write the plots to disk
    fig1.savefig(os.path.join("..", args['figures_dir'], 'brute_force_accuracy.png')
    fig2.savefig(os.path.join("..", args['figures_dir'], 'brute_force_loss.png')


# Run pipe with parameter hidden_layers and returns train and test accuracy for diagonal_search
def run_entire_pipeline_for_diagonal_search(**args):
    exp_data = pd.DataFrame(columns = COLUMNS)
    search_space = generate_search_space(args['ihls'], args['df'])
    reduced_search_space = diagonal_search(search_space)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for row in reduced_search_space:
        for model_architecture in row:
            df = model_architecture[0]
            ihls = model_architecture[1]
            args = parse_arguments([''])
            args['hidden_layers'] = calculate_model_architecture(df, ihls)
            train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, _, history = run_pipe(**args)
            # Accumulate plots for each model fits on a figure
            plot_accuracy_graphs_given_history(ax1, history, args['hidden_layers'])
            plot_loss_graphs_given_history(ax2, history, args['hidden_layers'])
            # Calculate sparsity score
            sparsity_score = calculate_sparsity_score(0.5, ihls, 11, df)
            # Append the current model results to csv
            current_row = pd.Series([train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, sparsity_score, args['hidden_layers']], index=COLUMNS)
            exp_data = exp_data.append(current_row, ignore_index=True)
            exp_data.to_csv(os.path.join("..", args['results_dir'], 'diagonal_search_results.csv'))
    # Write the plots to disk
    fig1.savefig(os.path.join("..", args['figures_dir'], 'diagonal_search_accuracy.png')
    fig2.savefig(os.path.join("..", args['figures_dir'], 'diagonal_search_loss.png')

# Run pipe with parameter hidden_layers and returns train and test accuracy for zigzag_search
sys.setrecursionlimit(100000)
"""
Zigzag search is a CPU-bound heavy computation and will raise  RecursionError: maximum recursion depth exceeded in comparison due to its heavy computations.
(Same for multithreading which does not exist in cpython due to GIL limitation).
To get more information about this limitation: https://stackoverflow.com/a/13592002/427887
"""
def run_entire_pipeline_for_zigzag_search(**args):
    exp_data = pd.DataFrame(columns = COLUMNS)
    exp_data.to_csv(os.path.join("..", args['results_dir'], 'zigzag_results.csv'))
    search_space = generate_search_space(args['ihls'], args['df'])
    result = zigzag_search(search_space)
    print(result) # by the time, the space is reduced all models will have already been ran in the pipeline due to ONLINE nature of this traversal.
    
# SEARCH and SPARSITY METHODS
# Task #1 
# Generate the search space for a binary classification problem
# # hidden layers and # nodes in those layers

def generate_search_space(max_x, max_y):
    """
    @params: max_x : x axis consists of integer multiples of 8. (each value in the range(8, max_x+1. 8) represents one ihls)
    @params: max_y : y axis just consists of powers of 2. (each value in the range 1,2,4,8,16,32 ... max_y (exclusive) represents one df)
    Each entry is the (division factor, initial hidden layer size)
    @returns search space (list)
    """
    search_space = []
    if max_x % 8 != 0 or max_y % 2 != 0: return search_space

    i = 1 
    while i < max_y:
        current_row = []
        for j in range(8, max_x+1, 8): 
            current_row.append((i, j))
        search_space.append(current_row)
        i *= 2
    return search_space

def calculate_model_architecture(df, ihls):
    """
    @params: df: division factor (int)
    @params: ihls: Initial hidden layer size (int)
    @returns: matrix: all hidden layer sizes (List)
    """
    matrix = []
    if ihls <0 or df<1: return matrix
    current_layer_size = ihls
    matrix.append(current_layer_size)
    while current_layer_size // df > 1 and df!=1:
        current_layer_size = current_layer_size // df
        matrix.append(current_layer_size)
    return matrix

def pretty_print_search_space(space):
    for i in range(len(space)):
        print(space[i], end="\n")


# Task 2
# Traverse the search space brute force search
def brute_force_search(space):
    return space

# Task 3
# Traverse the search space via diagonal rays
import collections
def diagonal_search(space):
    d = collections.defaultdict(list)
    for i in range(len(space)):
        for j in range(len(space[0])):
            if (i+j) %2 == 1: # collect alternating diagonals
                d[i+j].append(space[i][j])
    return d.values()


# Task 4
# Traverse the search space via zigzags
import collections
def zigzag_search(space):
    result = []
    highest_accuracy = float("-inf")
    model_architecture = (len(space) - 1,0) # pick bottom left corner because first zigzag will start with a secondary diagonal  
    result.append((highest_accuracy, model_architecture))
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    def traverse_one_zig_zag(space, start_x, start_y, isPrimary):
        if not (0<= start_x < len(space) and 0 <= start_y < len(space[0])) or get(space,start_x, start_y) == "#" : return
        # Collect all models in the diagonal
        elements_in_current_diagonal = generate_primary_diagonal(space, start_x,start_y) if isPrimary else generate_secondary_diagonal(space, start_x,start_y)
        highest_accuracy, current_architecture = result[0]
        # Iterative over all the models in the diagonal
        for (x, y) in elements_in_current_diagonal:
            # Already visited this element in another diagonal traversal. Skip it.
            if space[x][y] == "#": continue 
            args = parse_arguments([''])
            print("##############")
            print(x, y)
            pretty_print_search_space(space)
            print("##############")
            df = space[x][y][0]
            ihls = space[x][y][1]
            args['hidden_layers'] = calculate_model_architecture(df, ihls)
            # Write the results.             
            train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, _, history = run_pipe(**args)
            # Accumulate plots for each model fits on a figure
            plot_accuracy_graphs_given_history(ax1, history, args['hidden_layers'])
            plot_loss_graphs_given_history(ax2, history, args['hidden_layers'])
            # Calculate sparsity score
            sparsity_score = calculate_sparsity_score(0.5, ihls, 11, df)
            # Append the current model results to csv
            current_row = pd.Series([train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, sparsity_score, args['hidden_layers']], index=COLUMNS)
            exp_data = exp_data.append(current_row, ignore_index=True)
            exp_data.to_csv(os.path.join("..", args['results_dir'], 'zigzag_results.csv'), header=None, mode='a')
            
            # Update highest accuracy and the model
            if highest_accuracy < test_accuracy:
                current_architecture = (x,y)
                result[0] = (test_accuracy, current_architecture)
            # Mark the node as visited after traversing it
            space[x][y] = "#"
        
        # Current diagonal traversal is over, go over
        return traverse_one_zig_zag(space, current_architecture[0], current_architecture[1], not isPrimary)
            

    traverse_one_zig_zag(space ,model_architecture[0], model_architecture[1], False)     
    # Write the plots to disk
    fig1.savefig(os.path.join("..", args['figures_dir'], 'zigzag_search_accuracy.png')
    fig2.savefig(os.path.join("..", args['figures_dir'], 'zigzag_search_loss.png')
    return result

def get(space, i, j):
    if 0<=i<len(space) and 0<=j<len(space[0]) and space[i][j] != "#":
        return space[i][j]


def generate_primary_diagonal(space, i, j):
    """
    Diagonal will extend from top left corner to bottom right corner
    """
    result = set()
    def helper(space, i, j , result):
        if 0<=i<len(space) and 0<=j<len(space[0]) and (i,j) not in result:
            result.add((i,j))
            helper(space,i+1,j+1, result)
            helper(space,i-1,j-1, result)
    helper(space, i,j, result)
    return result

def generate_secondary_diagonal(space, i, j):
    """
    Diagonal will extend from bottom left corner to top right corner
    """
    result = set()
    def helper(space, i, j , result):
        if 0<=i<len(space) and 0<=j<len(space[0]) and (i,j) not in result:
            result.add((i,j))
            helper(space,i-1,j+1, result)
            helper(space,i+1,j-1, result)
    helper(space, i,j, result)
    return result


# Task 5
# Calculate measure of sparsity for a given model
def calculate_sparsity_score(alpha, ihls, ils, df):
    """
    @param: alpha: hyperparameter between 0 and 1. Closer to 1 means jumping factor is more weighted. Closer to 0 means the 1/df is more weighted.
    @param: ihls: Initial hidden layer size
    @param: ils: input layer size
    @param: df: division factor. (i.e. If first ihls is 40 and df is 2, then the hidden layers will have 40, 20, 10, 5, 2 sizes)
    @returns: sparsity score. Returns -1 if the alpha is outside of the [0,1] range.
    """
    if not 0<=alpha<=1: return -1
    jumping_factor = ihls/ils
    return round(alpha*(jumping_factor) + (1-alpha)*1/df, 4)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm           # import colormap stuff!
import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def generate_3d_height_map_given_csv_file():
    path_to_csv_file = os.path.join('..', 'results', 'titanic_brute_force_results.csv')

    if not os.path.exists(path_to_csv_file): 
        print("path to csv file doesn't exist")
        return 
    args = parse_arguments([''])
    df = pd.read_csv(path_to_csv_file)
    print("csv is read\n", df.head(5))
    # Get data
    _x = np.arange(8, args['ihls']+1, 8)
    _y = np.array([2**i for i in range(int(math.log(args['df'], 2)))])
    len_x = len(_x)
    len_y = len(_y)

    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    bottom = np.zeros_like(x)
    width = 1
    depth = 1

    # Setup the figure and axes
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.43)

    # Plot train accuracies
    train_accuracies = df['train_accuracy'].values.reshape((len_x * len_y)) 
    dz = train_accuracies
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax1.bar3d(x, y, bottom, width, depth, train_accuracies, color=rgba, zsort='average', shade=True)
    ax1.set_xlabel('Initial Hidden Layer Size')
    ax1.set_ylabel('Division Factor')
    ax1.set_title('Train Accuracies')

    # Plot test accuracies
    test_accuracies = df['test_accuracy'].values.reshape((len_x * len_y))
    dz = test_accuracies
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    ax2.bar3d(x, y, bottom, width, depth, test_accuracies, color=rgba, zsort='average', shade=True)
    ax2.set_xlabel('Initial Hidden Layer Size')
    ax2.set_ylabel('Division Factor')
    ax2.set_title('Test Accuracies')
    
    # Plot sparsity scores
    sparsity_scores = df['sparsity_score'].values.reshape((len_x * len_y))
    dz = sparsity_scores
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    ax3.bar3d(x, y, bottom, width, depth, sparsity_scores,  color=rgba, zsort='average', shade=True)
    ax3.set_xlabel('Initial Hidden Layer Size')
    ax3.set_ylabel('Division Factor')
    ax3.set_title('Sparsities')
    plt.show()

