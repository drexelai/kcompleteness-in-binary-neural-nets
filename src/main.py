from pipeline.argument_parser import parse_arguments
from pipeline.run_pipeline import run_entire_pipeline_for_brute_force_search
from pipeline.run_pipeline import run_entire_pipeline_for_diagonal_search
from pipeline.run_pipeline import run_entire_pipeline_for_zig_zag_search

import sys
import time

def main(args):
    args = parse_arguments(args)
    run_entire_pipeline_for_brute_force_search(**args)
    # run_entire_pipeline_for_diagonal_search(**args)
    # run_entire_pipeline_for_zig_zag_search(**args)


if __name__ == "__main__": 
    
    start_time = time.clock()
    main(sys.argv)
    print(time.clock() - start_time, "seconds")
