import argparse
import os

def create_empty_argument_parser():
	"""
	Create an empty argparse.ArgumentParser.
	"""
	import argparse
	return argparse.ArgumentParser(
		description='Setting up parser in sparsity paper',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
		allow_abbrev=False)

def add_necessary_arguments_to_parser():
	parser = create_empty_argument_parser()
	parser.add_argument('--batch_size', default=10, type=int, help='The batch size to train for.')
	parser.add_argument('--delta', default=50, type=int, help='The spacing between ni and nj when calculating the gradient of accuracy.')
	parser.add_argument('--early_stopping_patience', default=3, type=int, help='Number of epochs with no improvement after which training will be stopped if early stopping is enabled.')
	parser.add_argument('--epoch', default=15, type=int, help='The number of epochs to train for.')
	parser.add_argument('--results_dir', default="results", type=str, help='The directory of the experiment data is saved')
	parser.add_argument('--fig_save_dir', default="figures", type=str, help='The location where plots will be saved')
	parser.add_argument('--fig_save_name', default="figure", type=str, help='The name of the figure that is saved')
	parser.add_argument('--model_save_dir', default="models", type=str, help='The location where models will be saved')	
	parser.add_argument('--ihls', default=64, type=int, help='The dimension of initial the hidden layer.')
	parser.add_argument('--df', default=64, type=int, help='Maximum division factor')
	parser.add_argument('--no_early_stopping', default=False, action='store_true', help='Do not do early stopping.')
	parser.add_argument('--posterior_alpha', default=0.9, type=float, help='Probability threshold for sufficient evidence when evaluating posterior. Between 0 and 1')
	parser.add_argument('--test_size', default=0.2, type=float, help='The proportion of data used for tests. Between 0 and 1.')
	parser.add_argument('--threshold', default=0.5, type=float, help='The cutoff to predict 1 vs 0. Between 0 and 1.')
	parser.add_argument('--use_churn_data', default=True, action='store_true', help='Use churn model data.')
	parser.add_argument('--use_titanic_data', default=False, action='store_true', help='Use titanic data.')
	parser.add_argument('--validation_split', default=0.2, type=float, help='Validation split proportion while training. Between 0 and 1.')
	parser.add_argument('--verbose', default=1, type=int, help='Verbose for training. Either 0, 1, or 2.')
	return parser


def parse_arguments(args):
    parser = add_necessary_arguments_to_parser()
    args, _ = parser.parse_known_args(args)
    args = vars(args)
    return args
