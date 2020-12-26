# Author: Isamu Isozaki, Yigit Alparslan
# Date: 2020/11/10
# Purpose: Train and test for models with hidden layer at n dimensions
from get_data import get_titanic_data, get_data_churn_rate
from train import train
from test import test
from argument_parser import parse_arguments
from search_sparsity import generate_search_space
from search_sparsity import brute_force_search
from search_sparsity import diagonal_search
from search_sparsity import zigzag_search
from search_sparsity import calculate_model_architecture
import pandas as pd
# Run pipe and returns train accuracy and test accuracy
def run_pipe(**args):
    prepare_data = get_data_churn_rate if args["use_churn_data"] else get_titanic_data
    X_train, X_test, y_train, y_test = prepare_data(**args)
    model, train_accuracy, val_accuracy = train(X_train, y_train, **args)
    test_accuracy, area_under_curve, precision, recall, F1 = test(X_test, y_test, model, **args)
    return train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model

# Run pipe with parameter hidden_layers and returns train and test accuracy for brute_force
def run_entire_pipeline_for_brute_force_search(**args):
    columns = [
		'train_accuracy', 
		'val_accuracy', 
		'test_accuracy', 
		'area_under_curve', 
		'precision',
		'recall',
		'F1']
    exp_data = pd.DataFrame(columns = columns)

    search_space = generate_search_space(args['ihls'], args['df'])
    reduced_search_space = brute_force_search(search_space)
    for model_architecture in reduced_search_space:
        args['hidden_layers'] = calculate_model_architecture(*model_architecture)
        train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model = run_pipe(**args)
        current_row = pd.DataFrame([train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1], columns=columns)
        exp_data = exp_data.append(current_row)
        exp_data.to_csv(os.path.join(args['results_dir'], 'brute_force_results.csv'))


# Run pipe with parameter hidden_layers and returns train and test accuracy for brute_force
def run_entire_pipeline_for_diagonal_search(**args):
    columns = [
		'train_accuracy', 
		'val_accuracy', 
		'test_accuracy', 
		'area_under_curve', 
		'precision',
		'recall',
		'F1']
    exp_data = pd.DataFrame(columns = columns)

    search_space = generate_search_space(args['ihls'], args['df'])
    reduced_search_space = diagonal_search(search_space)
    for model_architecture in reduced_search_space:
        args['hidden_layers'] = calculate_model_architecture(*model_architecture)
        train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model = run_pipe(**args)
        current_row = pd.DataFrame([train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1], columns=columns)
        exp_data = exp_data.append(current_row)
        exp_data.to_csv(os.path.join(args['results_dir'], 'diagonal_search.csv'))


# Run pipe with parameter hidden_layers and returns train and test accuracy for brute_force
def run_entire_pipeline_for_zig_zag_search(**args):
    # TODO
    columns = [
		'train_accuracy', 
		'val_accuracy', 
		'test_accuracy', 
		'area_under_curve', 
		'precision',
		'recall',
		'F1']
    exp_data = pd.DataFrame(columns = columns)

    search_space = generate_search_space(args['ihls'], args['df'])
    reduced_search_space = zigzag_search(search_space)
    for model_architecture in reduced_search_space:
        args['hidden_layers'] = calculate_model_architecture(*model_architecture)
        train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1, model = run_pipe(**args)
        current_row = pd.DataFrame([train_accuracy, val_accuracy, test_accuracy, area_under_curve, precision, recall, F1], columns=columns)
        exp_data = exp_data.append(current_row)
        exp_data.to_csv(os.path.join(args['results_dir'], 'zigzag.csv'))


