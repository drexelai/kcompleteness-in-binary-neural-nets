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
    # TODO: 
    result = []
    for (i,j) in generate_one_zig(space, len(space)-1,0):
        print(i,j)
    # for i in range()
    # def dfs(space,i,j): 
    #     pass

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


if __name__ == "__main__": 
    search_space = generate_search_space(48, 64)
    print(generate_secondary_diagonal(search_space, 4, 3))
