from yens_k_shortest_paths_finder import YensGraph
from grasp_disjoint_paths import GRASPGraph
import time
import random
import csv

# Dane grafu
graph_data = [
    (1, 2, 3),
    (1, 3, 2),
    (2, 3, 1),
    (2, 4, 5),
    (3, 4, 4),
    (3, 5, 2),
    (4, 5, 1),
    (4, 6, 3),
    (5, 6, 2),
    (5, 7, 4),
    (6, 7, 5),
    (6, 8, 2),
    (7, 8, 1),
    (7, 9, 3),
    (8, 9, 4),
    (8, 10, 5),
    (9, 10, 2),
]

def run_test(algorithm, num_nodes, start_node, end_node, k_paths):
    # Sprawdzenie, czy węzły istnieją w grafie
    if start_node < 1 or start_node > num_nodes or end_node < 1 or end_node > num_nodes:
        raise ValueError("Węzeł startowy lub końcowy jest poza zakresem.")

    start_time = time.time()
    paths = algorithm(start_node, end_node, k_paths)
    end_time = time.time()

    if paths is None:
        return end_time - start_time, []

    return end_time - start_time, paths

def apply_algorithms_to_graph(graph_data, algorithms):
    for u, v, weight in graph_data:
        for algorithm in algorithms:
            algorithm.add_edge(u, v, weight)

def run_benchmarks(num_tests, k_paths):
    results = []
    for num_nodes in range(5, num_tests + 1, 5):
        num_edges = min(num_nodes * (num_nodes - 1) // 2, num_nodes * 3)
        yens_graph = YensGraph()
        grasp_graph = GRASPGraph()
        apply_algorithms_to_graph(graph_data, (yens_graph, grasp_graph))

        start_node = 1
        end_node = num_nodes

        yen_time, path_y = run_test(yens_graph.yen, num_nodes, start_node, end_node, k_paths)
        grasp_time, path_g = run_test(grasp_graph.grasp, num_nodes, start_node, end_node, k_paths)


        if path_y:
            print(path_y, path_g)
            print("####")
            results.append((num_nodes, num_edges, yen_time, grasp_time))

    return results

def export_benchmark_results_to_csv(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Number of Nodes', 'Number of Edges', "Yen's Execution Time (s)", 'GRASP Execution Time (s)'])
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    num_tests = 700
    k_paths = 50
    max_attempts = 3

    benchmark_results = run_benchmarks(num_tests, k_paths)
    export_benchmark_results_to_csv(benchmark_results, 'benchmark_results.csv')
