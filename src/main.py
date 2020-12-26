from pipeline import run_entire_pipeline
from argument_parser import parse_arguments
from pipeline import run_entire_pipeline_for_brute_force_search
from pipeline import run_entire_pipeline_for_diagonal_search
from pipeline import run_entire_pipeline_for_zig_zag_search

import sys
def main(args):
    args = parse_arguments(args)
    run_entire_pipeline_for_brute_force_search(**args)
    run_entire_pipeline_for_diagonal_search(**args)
    run_entire_pipeline_for_zig_zag_search(**args)


if __name__ == "__main__": 
    # search_space = generate_search_space(48, 64)
    # print(zigzag_search(search_space))
    main(sys.argv)

