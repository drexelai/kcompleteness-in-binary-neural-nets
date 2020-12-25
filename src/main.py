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

def calculate_model_architecture(ihls, df):
    """
    @params: ihls: Initial hidden layer size
    @params: df: division factor
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
    d = collections.defaultdict(list)
    for i in range(len(space)):
        for j in range(len(space[0])):
            if (i+j) %2 == 1: # collect alternating diagonals
                d[i+j].append(space[i][j])
    return d.values()
            

if __name__ == "__main__": 
    space = generate_search_space(48, 64)
    print(diagonal_search(space))
