from pipeline.argument_parser import parse_arguments
from pipeline.run_pipeline import run_entire_pipeline_for_brute_force_search
from pipeline.run_pipeline import run_entire_pipeline_for_diagonal_search
from pipeline.run_pipeline import run_entire_pipeline_for_zigzag_search
from pipeline.run_pipeline import generate_search_space, pretty_print_search_space, zigzag_search

import sys
import time

def main(args):
    args = parse_arguments(args)
    # run_entire_pipeline_for_brute_force_search(**args) # 1178.3585526 seconds with df = 8, ihls = 48
    # run_entire_pipeline_for_diagonal_search(**args) # 530.6440734 seconds with df = 8, ihls = 48
    run_entire_pipeline_for_zigzag_search(**args)

if __name__ == "__main__": 
    
    start_time = time.clock()
    main(sys.argv)
    print(time.clock() - start_time, "seconds")
