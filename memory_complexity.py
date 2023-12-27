import sys
import matplotlib.pyplot as plt
import random
from grasp_disjoint_paths import GRASPGraph
from yens_k_shortest_paths_finder import YensGraph
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np


def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def run_grasp(graph, start_node, end_node, k_paths):
    return graph.yen(start_node, end_node, k_paths)


def generate_deterministic_graph(num_nodes, additional_edges_per_node, weight_range):
    edges = []
    for i in range(1, num_nodes):
        for j in range(i + 1, min(i + 1 + additional_edges_per_node, num_nodes + 1)):
            weight = random.uniform(*weight_range)
            edges.append((i, j, weight))
    return edges


def plot_memory_complexity(input_sizes, memory_usage):
    sns.set_theme(style='whitegrid', palette='pastel')
    plt.figure(figsize=(12, 6))

    plt.errorbar(input_sizes, memory_usage, fmt='-o', color='darkblue', 
                 ecolor='lightblue', elinewidth=3, capsize=5, capthick=2, markersize=8)

    # Adding grid
    plt.grid(True, which='major', linestyle='--', linewidth=0.5) 

    # Adding title and axis labels
    plt.title("Złożoność pamięciowa algorytmu Yen'a", fontsize=16)
    plt.xlabel("Liczba węzłów w grafie", fontsize=14)
    plt.ylabel("Zużycie pamięci (KB)", fontsize=14)

    # Formatting the axis ticks
    plt.xticks(np.arange(min(input_sizes), max(input_sizes)+1, step=5), fontsize=12)
    # Determine a reasonable interval for y-ticks based on the memory usage data
    y_interval = max(0.05, round((max(memory_usage) - min(memory_usage)) / 10, 2))
    plt.yticks(np.arange(0, max(memory_usage)+y_interval, step=y_interval), fontsize=12)

    # Set y-axis formatter for better label precision
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Limiting the axes to add some padding around the data for better visibility
    plt.xlim([min(input_sizes) - 1, max(input_sizes) + 1])
    plt.ylim([0, max(memory_usage) + y_interval])

    plt.tight_layout()  # Adjust subplot params to fit into the figure area

    # Display the plot
    plt.show()


def main():
    NUM_NODES_LIST = range(5, 100, 5)
    K_PATHS = 5
    ITERATIONS_PER_SIZE = 20
    ADDITIONAL_EDGES_PER_NODE = 2
    WEIGHT_RANGE = (1, 15)

    memory_usage = []

    for num_nodes in NUM_NODES_LIST:
        total_memory = 0
        for _ in range(ITERATIONS_PER_SIZE):
            graph = YensGraph()
            graph_edges = generate_deterministic_graph(
                num_nodes, ADDITIONAL_EDGES_PER_NODE, WEIGHT_RANGE
            )
            for u, v, w in graph_edges:
                graph.add_edge(u, v, w)
            result = run_grasp(graph, 1, num_nodes, K_PATHS)
            total_memory += get_size(result)
        total_memory = total_memory / 1024
        # Oblicz średnie zużycie pamięci dla danego rozmiaru grafu
        average_memory = total_memory / ITERATIONS_PER_SIZE
        memory_usage.append(average_memory)

    plot_memory_complexity(NUM_NODES_LIST, memory_usage)


if __name__ == "__main__":
    main()
