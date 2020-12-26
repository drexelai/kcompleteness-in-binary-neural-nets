from pipeline import run_pipe
from argument_parser import parse_arguments
# Task #1 
# Generate the search space for a binary classification problem
# # hidden layers and # nodes in those layers

def generate_search_space(max_x, max_y):
    """
    @params: max_x : x axis consists of integer multiples of 8.
    @params: max_y : y axis just consists of powers of 2.
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
    while current_layer_size // df > 1:
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
    # TODO: 
    result = []
    highest_accuracy = float("-inf")
    model_architecture = (len(space) - 1,0) # pick bottom left corner because first zigzag will start with a secondary diagonal  
    result.append((highest_accuracy, model_architecture))
    
    def traverse_one_zig_zag(space, start_x, start_y, isPrimary):
        if not (0<= start_x < len(space) and 0 <= start_y < len(space[0])) or get(space,start_x, start_y) == "#" : return
        # Collect all models in the diagonal
        models_in_current_diagonal = generate_primary_diagonal(space, start_x,start_y) if isPrimary else generate_secondary_diagonal(space, start_x,start_y)
        highest_accuracy, current_architecture = result[0]
        # Iterative over all the models in the diagonal
        for (df,ihls) in models_in_current_diagonal:
            args = parse_arguments([''])
            args['hidden_layers'] = calculate_model_architecture(df, ihls)
            _, _, current_accuracy, _, _, _, _, _  = run_pipe(**arg)
            # Update highest accuracy and the model
            if highest_accuracy < current_accuracy:
                result[0] = (current_accuracy, (df,ihls))
            # Mark the node as visited after traversing it
            space[df][ihls] = "#"
        
        # Current diagonal traversal is over, go over
        traverse_one_zig_zag(space, current_architecture[0], current_architecture[1], not isPrimary)
            

    traverse_one_zig_zag(space ,model_architecture[0], model_architecture[1], False)      
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
