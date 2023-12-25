from grasp_disjoint_paths import GRASPGraph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import csv


def run_grasp_and_measure_time(graph, start_node, end_node, k_paths):
    start_time = time.time()
    paths = graph.grasp(start_node, end_node, k_paths)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time, paths


def generate_deterministic_graph(num_nodes, additional_edges_per_node, weight_range):
    edges = []
    for i in range(1, num_nodes):
        for j in range(i + 1, min(i + 1 + additional_edges_per_node, num_nodes + 1)):
            weight = random.uniform(*weight_range)
            edges.append((i, j, weight))
    return edges


def export_graph_and_paths_to_csv(graph_edges, paths, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Graph Edges", "Paths"])
        for edges, path_set in zip(graph_edges, paths):
            edges_str = "; ".join([f"{u}-{v}-{w}" for u, v, w in edges])
            paths_str = "; ".join([str(path) for path in path_set])
            writer.writerow([edges_str, paths_str])

# def plot_results(num_nodes_list, average_times, std_dev_times):
#     sns.set(style='whitegrid', palette='muted')
#     plt.figure(figsize=(14, 8))
#     plt.gca().set_facecolor('#f0f0f0')

#     plt.errorbar(num_nodes_list, average_times, yerr=std_dev_times, fmt='-o', 
#                  color='midnightblue', ecolor='skyblue', elinewidth=3, capsize=5, 
#                  capthick=3, markersize=10, label='GRASP Algorithm')
#     plt.gca().get_lines()[0].set_alpha(0.7)

#     plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
#     plt.title('Złożoność czasowa algorytmu GRASP', fontsize=24, fontweight='bold')
#     plt.xlabel('Liczba węzłów w grafie', fontsize=18, fontweight='bold')
#     plt.ylabel('Średni czas wykonania (s)', fontsize=18, fontweight='bold')
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.xticks(np.arange(min(num_nodes_list), max(num_nodes_list)+1, step=10), fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylim(bottom=min(average_times)-min(std_dev_times), top=max(average_times)+max(std_dev_times))

#     legend = plt.legend(fontsize=14, shadow=True)
#     frame = legend.get_frame()
#     frame.set_color('white')
#     frame.set_edgecolor('black')

#     plt.tight_layout()
#     plt.show()


def plot_results(num_nodes_list, average_times, std_dev_times):
    sns.set_theme(style='whitegrid', palette='pastel')
    plt.figure(figsize=(12, 6))

    # Create the error bar plot
    plt.errorbar(num_nodes_list, average_times, yerr=std_dev_times, fmt='-o', color='darkblue', 
                ecolor='lightblue', elinewidth=3, capsize=5, capthick=2, markersize=8)
    
    plt.grid(True, which='major', linestyle='--', linewidth=0.5) 
    
    # Title and labels with improved font sizes
    plt.title('Złożoność czasowa algorytmu Yen\'a', fontsize=16)
    plt.xlabel('Liczba węzłów w grafie', fontsize=14)
    plt.ylabel('Średni czas wykonania (s)', fontsize=14)

    # Customizing tick sizes
    plt.xticks(np.arange(min(num_nodes_list), max(num_nodes_list)+1, 5.0), fontsize=12)
    plt.yticks(fontsize=12)

    # Layout adjustments
    plt.tight_layout()

    # Show the plot
    plt.show()

def main():
    """Main function to execute the graph analysis and export results."""
    # Constants
    NUM_NODES_LIST = range(5, 100, 5)
    K_PATHS = 5
    ITERATIONS_PER_SIZE = 20
    ADDITIONAL_EDGES_PER_NODE = 2
    WEIGHT_RANGE = (1, 15)

    average_times = []
    std_dev_times = []
    all_graph_edges = []
    all_paths = []

    for num_nodes in NUM_NODES_LIST:
        iteration_times = []
        for _ in range(ITERATIONS_PER_SIZE):
            graph = GRASPGraph()
            graph_edges = generate_deterministic_graph(
                num_nodes, ADDITIONAL_EDGES_PER_NODE, WEIGHT_RANGE
            )
            for u, v, w in graph_edges:
                graph.add_edge(u, v, w)
            elapsed_time, paths = run_grasp_and_measure_time(graph, 1, num_nodes, K_PATHS)
            iteration_times.append(elapsed_time)
            all_graph_edges.append(graph_edges)
            all_paths.append(paths)

        average_times.append(np.mean(iteration_times))
        std_dev_times.append(np.std(iteration_times))

    

    export_graph_and_paths_to_csv(all_graph_edges, all_paths, "grasp_graph_paths.csv")
    plot_results(NUM_NODES_LIST, average_times, std_dev_times)

if __name__ == "__main__":
    main()
