"""
Finds the optimal graphs for each duration set
given the largest path length N in the duration set
"""
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def show_graph_with_labels(filename, title, adjacency_matrix):
    """
    Takes a numpy adjacency matrix and graphs it as a directed graph
    Saves the graph with the title to the filename given
    """
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    axes = plt.gca()
    axes.set_title(title)
    nx.draw(
        graph,
        node_size=400,
        connectionstyle='arc3, rad = 0.2',
        with_labels=True,
        labels={n: n + 1 for n in graph.nodes},
        pos=nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot", args="-Grankdir=LR"),
        ax=axes
    )
    plt.savefig(filename + ".png")
    print("Saved " + filename + ".png")
    plt.clf()


def check_rows_valid(N, combination):
    for i in range(1, N - 2):
        row_sum = 0
        for col_starter in col_starters[i - 1:]:
            row_sum += combination[col_starter + i]
        if row_sum == 0:
            return False
    return True


def find_durations(matrix):
    """
    Finds the duration set of the matrix by taking it's powers
    The number of paths from source to target can be found in the top right entry of the matrix
    """
    duration_set = {}
    for power in range(1, len(matrix)):
        amount = int(np.linalg.matrix_power(matrix, power)[0][-1])
        if amount > 0:
            duration_set[power] = amount
    return duration_set

def generate_matrix_from_combination(N, combination):
    """
    Returns the numpy N x N matrix if successful
    Otherwise returns None
    Counts is an array passed by reference, the 2- and 3- indices are changed
    """
    # check if rows are valid
    if not check_rows_valid(N, combination):
        return None

    # initialize the first two columns
    first_col = np.zeros(N)
    second_col = np.zeros(N)
    second_col[0] = 1
    columns = [first_col, second_col]

    # generate the remaining columns and check if they're valid
    for col_index in range(len(col_starters) - 1):
        column = np.zeros(N)
        data = combination[col_starters[col_index]: col_starters[col_index + 1]]
        # check validity of cols
        if sum(data) == 0:
            return None
        column[0: len(data)] = data
        columns.append(column)

    # append last col, don't do checks on this one
    last_col = np.zeros(N)
    last_col[0: N - 2] = combination[col_starters[-1]:]
    last_col[N - 2] = 1
    columns.append(last_col)

    # turn the columns into a matrix
    matrix = np.column_stack(columns)
    return matrix


if __name__ == "__main__":
    N = 8
    best_lexicographical = {}

    # store the start of the columns for checking if graph is valid
    col_starters = [0]
    next_col = 0
    for gap in range(2, N - 1):
        next_col += gap
        col_starters.append(next_col)

    num_positions = int(N * (N - 1) / 2) - 2

    counts = [0, 0, 0, 0]
    for combination in itertools.product([0, 1], repeat=num_positions):
        num_edges = sum(combination) + 2
        if (num_edges > (2 * N) - 3):
            counts[1] += 1

        matrix = generate_matrix_from_combination(N, combination)
        if matrix is None:
            counts[2] += 1
            continue

        # find the number of vertices, edges and paths
        durations = find_durations(matrix)
        num_paths = sum(durations.values())
        duration_set = set(durations.keys())

        # from comparison with triangle, any with N nodes but a max duration less than N-1 is bad
        if (N - 1) not in duration_set:
            counts[3] += 1
            continue

        print("num paths: {}, num vertices: {}, num edges: {}".format(num_paths, N, num_edges))

        duration_tuple = tuple(sorted(list(durations.keys())))
        if best_lexicographical.get(duration_tuple) is None:
            best_lexicographical[duration_tuple] = (num_paths, N, num_edges, matrix)
        else:
            prev_best = best_lexicographical.get(duration_tuple)
            if num_paths < prev_best[0] or (num_paths == prev_best[0] and num_edges < prev_best[2]):
                best_lexicographical[duration_tuple] = (num_paths, N, num_edges, matrix)

        '''
        filename = str(num_paths) + "-" + str(N) + "-" + str(num_edges) + "_" + str(duration_set) + "_" + "Graph" + str(counts[0])
        title = filename.replace("_", " ") + " " + str(combination)

        # plot the graph and save it
        if counts[0] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 10, 21, 52, 53]:
            show_graph_with_labels(filename, title, matrix)
        '''

        counts[0] += 1

    for key, value in best_lexicographical.items():
        name = str(value[0]) + "-" + str(value[1]) + "-" + str(value[2]) + "_" + str(key)
        show_graph_with_labels(name, name, value[3])

    print("Count: " + str(counts))
