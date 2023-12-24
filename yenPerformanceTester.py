from yens_k_shortest_paths_finder import YensGraph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time


def run_yen_and_measure_time(graph, start_node, end_node, k_paths):
    start_time = time.time()
    paths = graph.yen(start_node, end_node, k_paths)
    end_time = time.time()
    return end_time - start_time

def generate_deterministic_graph(num_nodes, additional_edges_per_node, weight_range):
    edges = []
    for i in range(1, num_nodes):
        for j in range(i+1, min(i+1+additional_edges_per_node, num_nodes+1)):
            weight = random.uniform(*weight_range)
            edges.append((i, j, weight))
    return edges

# Ustawienia do symulacji
num_nodes_list = range(5, 100, 5)  # Liczba węzłów w grafie
k_paths = 3  # Liczba ścieżek do znalezienia
iterations_per_size = 20  # Liczba iteracji dla każdego rozmiaru grafu
additional_edges_per_node = 2  # Dodatkowe krawędzie dla każdego nowego węzła
weight_range = (1, 15)  # Zakres wag krawędzi

# Zbieranie danych do analizy
average_times = []
std_dev_times = []

for num_nodes in num_nodes_list:
    iteration_times = []
    for _ in range(iterations_per_size):
        graph = YensGraph()
        graph_edges = generate_deterministic_graph(num_nodes, additional_edges_per_node, weight_range)
        # Symulacja dodawania krawędzi do grafu - ta część musi być zaimplementowana w prawdziwym kodzie
        for u, v, w in graph_edges:
            graph.add_edge(u, v, w)
        elapsed_time = run_yen_and_measure_time(graph, 1, num_nodes, k_paths)
        iteration_times.append(elapsed_time)
    average_times.append(np.mean(iteration_times))
    std_dev_times.append(np.std(iteration_times))


# Tworzenie wykresu
sns.set_theme(style='whitegrid', palette='pastel')
plt.figure(figsize=(12, 6))

# Create the error bar plot
plt.errorbar(num_nodes_list, average_times, yerr=std_dev_times, fmt='-o', color='darkblue', 
             ecolor='lightblue', elinewidth=3, capsize=5, capthick=2, markersize=8)

# Title and labels with improved font sizes
plt.title('Złożoność czasowa algorytmu Yen\'a', fontsize=16)
plt.xlabel('Liczba węzłów w grafie', fontsize=14)
plt.ylabel('Średni czas wykonania (s)', fontsize=14)

# Customizing tick sizes
plt.xticks(np.arange(min(num_nodes_list), max(num_nodes_list)+1, 5.0), fontsize=12)
plt.yticks(np.arange(0, max(average_times)+0.05, 0.05), fontsize=12)

# Layout adjustments
plt.tight_layout()

# Show the plot
plt.show()