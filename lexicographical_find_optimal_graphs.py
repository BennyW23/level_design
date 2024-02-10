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


def find_durations(N, matrix):
    """
    Finds the duration set of the matrix by taking it's powers
    The number of paths from source to target can be found in the top right entry of the matrix
    """
    paths = [0] * N
    paths[0] = 1
    durations = [set()] * N
    durations[0] = {0}
    for src in range(0, N - 1):
        row = matrix[src]
        for target in range(src + 1, N):
            if row[target] == 1:
                paths[target] += paths[src]
                durations[target] = durations[target].union(set(map(lambda x: x + 1, durations[src])))
    return (paths[-1], durations[-1])


def generate_matrix_from_array(N, array):
    """
    Returns the numpy N x N matrix if successful
    Otherwise returns None
    Counts is an array passed by reference, the 2- and 3- indices are changed
    """
    # initialize the first two columns
    first_col = np.zeros(N, int)
    second_col = np.zeros(N, int)
    second_col[0] = 1
    columns = [first_col, second_col]

    idx = 0
    # generate the remaining columns and check if they're valid
    for col_index in range(1, N - 1):
        new_idx = idx + col_index
        column = np.zeros(N, int)

        data = array[idx: new_idx]
        column[0: col_index] = data
        column[col_index] = 1
        columns.append(column)
        idx = new_idx

    # turn the columns into a matrix
    matrix = np.column_stack(columns)
    return matrix


def generate_array_from_combination(size, combination):
    arr = np.zeros(size, int)
    for index in combination:
        arr[index] = 1
    return arr


def find_best_lexicographical(N):
    best_lexicographical = {}

    # store the start of the columns for checking if graph is valid
    col_starters = [0]
    next_col = 0
    for gap in range(2, N - 1):
        next_col += gap
        col_starters.append(next_col)

    num_positions = int((N - 1) * (N - 2) / 2)

    counts = [0, 0]
    for num_ones in range(0, N - 1):
        for combination in itertools.combinations(range(num_positions), num_ones):
            flattened = generate_array_from_combination(num_positions, combination)
            matrix = generate_matrix_from_array(N, flattened)

            # find the number of vertices, edges and paths
            num_paths, duration_set = find_durations(N, matrix)
            if num_paths > len(duration_set):
                counts[1] += 1
                continue
            num_edges = num_ones + N - 1

            print("num paths: {}, num vertices: {}, num edges: {}".format(num_paths, N, num_edges))
            counts[0] += 1

            duration_tuple = tuple(sorted(duration_set))
            if best_lexicographical.get(duration_tuple) is None:
                best_lexicographical[duration_tuple] = (num_paths, N, num_edges, matrix)
            else:
                prev_best = best_lexicographical.get(duration_tuple)
                if num_edges < prev_best[2]:
                    best_lexicographical[duration_tuple] = (num_paths, N, num_edges, matrix)

            '''
            filename = str(num_paths) + "-" + str(N) + "-" + str(num_edges) + "_" + str(duration_set) + "_" + "Graph" + str(counts[0])
            title = filename.replace("_", " ") + " " + str(combination)

            # plot the graph and save it
            show_graph_with_labels(filename, title, matrix)
            '''

    for key, value in best_lexicographical.items():
        name = str(value[0]) + "-" + str(value[1]) + "-" + str(value[2]) + "_" + str(key)
        print(name)
        show_graph_with_labels(name, name, value[3])

    print("Count: " + str(counts))


def find_best_with_duration(duration):
    N = max(duration) + 1
    # store the start of the columns for checking if graph is valid
    col_starters = [0]
    next_col = 0
    for gap in range(2, N - 1):
        next_col += gap
        col_starters.append(next_col)

    num_positions = int((N - 1) * (N - 2) / 2)
    best_edges = np.inf

    full_set = set(range(1, N))
    # count = 0
    for num_ones in range(5, N - 1):
        print(num_ones)
        num_edges = num_ones + N - 1
        if num_edges > best_edges:
            break
        for combination in itertools.combinations(range(num_positions), num_ones):
            flattened = generate_array_from_combination(num_positions, combination)
            matrix = generate_matrix_from_array(N, flattened)

            # find the number of vertices, edges and paths
            num_paths, duration_set = find_durations(N, matrix)
            if duration_set != full_set:
                continue
            if num_paths > len(full_set):
                continue
            if num_edges < best_edges:
                best_edges = num_edges

            print("num paths: {}, num vertices: {}, num edges: {}".format(num_paths, N, num_edges))

            filename = str(num_paths) + "-" + str(N) + "-" + str(num_edges) + "_" + str(flattened)
            title = f"{num_paths} paths, {N} vertices, {num_edges} edges"

            # plot the graph and save it
            show_graph_with_labels(filename, title, matrix)
            # count += 1
    # print(count)


if __name__ == "__main__":
    durations = list(range(1,11))
    find_best_with_duration(set(durations))
