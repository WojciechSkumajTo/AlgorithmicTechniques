from yens_k_shortest_paths_finder import YensGraph
from grasp_disjoint_paths import GRASPGraph
import networkx as nx
import time
import random
import csv


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


def generate_graph_data(num_nodes, num_edges, weight_range):
    if num_edges > num_nodes * (num_nodes - 1) // 2:
        raise ValueError(f"Zbyt wiele krawędzi ({num_edges}) dla podanej liczby węzłów ({num_nodes}).")
    if num_edges < num_nodes - 1:
        raise ValueError(f"Zbyt mało krawędzi ({num_edges}) dla utworzenia spójnego grafu z {num_nodes} węzłami.")

    # Tworzenie szkieletu grafu w formie drzewa rozpinającego
    edges = [(i, i + 1) for i in range(1, num_nodes)]
    num_edges -= len(edges)

    # Dodawanie pozostałych krawędzi
    all_possible_edges = [(i, j) for i in range(1, num_nodes + 1) for j in range(i + 2, num_nodes + 1)]
    edges += random.sample(all_possible_edges, min(num_edges, len(all_possible_edges)))

    # Przypisywanie losowych wag
    return [(u, v, random.randint(*weight_range)) for u, v in edges]




def apply_algorithms_to_graph(graph_data, algorithms):
    for u, v, weight in graph_data:
        for algorithm in algorithms:
            algorithm(u, v, weight)


def run_benchmarks(num_tests, k_paths, max_attempts=3):
    results = []
    for num_nodes in range(2, num_tests + 1, 3):
        attempt = 0
        while attempt < max_attempts:
            try:
                num_edges = random.randint(num_nodes - 1, min(num_nodes * (num_nodes - 1) // 2, num_nodes * 3))
                graph_data = generate_graph_data(num_nodes, num_edges, (1, 50))

                yens_graph = YensGraph()
                grasp_graph = GRASPGraph()
                apply_algorithms_to_graph(graph_data, (yens_graph.add_edge, grasp_graph.add_edge))

                start_node = 1
                end_node = num_nodes

                yen_time, path_y = run_test(yens_graph.yen, num_nodes, start_node, end_node, k_paths) #! dodałem sciezki
                grasp_time, path_g = run_test(grasp_graph.grasp, num_nodes, start_node, end_node, k_paths)
                
                results.append((num_nodes, num_edges, yen_time, grasp_time))     
                break  # Wyjdź z pętli, jeśli graf został pomyślnie wygenerowany i przetestowany
            
            except ValueError as e:
                print(f"Błąd podczas generowania grafu: {e}")
                attempt += 1

        if attempt == max_attempts:
            print(f"Nie udało się wygenerować poprawnego grafu po {max_attempts} próbach dla {num_nodes} węzłów.")

    return results

def export_benchmark_results_to_csv(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Number of Nodes', 'Number of Edges', "Yen's Execution Time (s)", 'GRASP Execution Time (s)'])
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":

    num_tests = 100  # Liczba różnych wielkości grafów do przetestowania
    k_paths = 3  # Liczba ścieżek do znalezienia

    benchmark_results = run_benchmarks(num_tests, k_paths)
    export_benchmark_results_to_csv(benchmark_results, 'benchmark_results.csv')


