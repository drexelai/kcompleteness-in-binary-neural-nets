from pipeline.argument_parser import parse_arguments
from pipeline.run_pipeline import run_entire_pipeline_for_brute_force_search
from pipeline.run_pipeline import run_entire_pipeline_for_diagonal_search
from pipeline.run_pipeline import run_entire_pipeline_for_zigzag_search
from pipeline.run_pipeline import generate_search_space, pretty_print_search_space, zigzag_search, generate_3d_height_map_given_csv_file
import sys
import time
import os
def main(args):
    args = parse_arguments(args)
    # Churn model: Some runs for the brute force pipeline
    # 1178.3585526 seconds with df = 8, ihls = 48
    # 3854.4821924 seconds with df = 64 and ihls = 64
    # Titanic model: Some runs for the brute force pipeline
    # 139.0985318 seconds with df = 64 and ihls = 64
    # run_entire_pipeline_for_brute_force_search(**args) 
    
    # Churn model: Some runs for the diagonal search pipeline
    # 530.6440734 seconds with df = 8, ihls = 48
    # 1989.9744428 seconds with df = 64 and ihls = 64
    # Titanic model: Some runs for diagonal search pipeline
    # 78.695224 seconds with df = 64 and ihls = 64
    # run_entire_pipeline_for_diagonal_search(**args) 
    
    # Churn model: Some runs for the zigzag search pipeline
    # 240.3445642 seconds with df = 64, ihls = 48
    # 11 min 12 sec = 672 seconds with df = 64 and ihls = 64
    # Titanic model: Some runs for diagonal search pipeline
    # 66.6 seconds with df = 64 and ihgls = 64
    run_entire_pipeline_for_zigzag_search(**args) 

    # Plotting purposes
    # generate_3d_height_map_given_csv_file()

if __name__ == "__main__": 
    
    start_time = time.clock()
    main(sys.argv)
    print(time.clock() - start_time, "seconds")
