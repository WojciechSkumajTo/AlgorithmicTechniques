import random
import heapq
import networkx as nx

class GRASPGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_edge(self, from_node, to_node, weight):
        self.graph.add_edge(from_node, to_node, weight=weight)

    def greedy_randomized_construction(self, start, end):
        path = [start]
        current = start
        while current != end:
            if current not in self.graph:
                return []  # Zabezpieczenie przed brakiem węzła w grafie
            neighbors = list(self.graph.successors(current))
            if not neighbors:
                return []  # Brak dalszej ścieżki
            next_node = random.choice(neighbors)
            path.append(next_node)
            current = next_node
        return path if current == end else []

    def local_search(self, path):
        improved = False
        for i in range(1, len(path) - 1):
            for neighbor in self.graph.successors(path[i - 1]):
                if neighbor != path[i] and neighbor in self.graph:
                    if path[i + 1] in self.graph[neighbor]:
                        new_weight = (
                            self.graph[path[i - 1]][neighbor]["weight"]
                            + self.graph[neighbor][path[i + 1]]["weight"]
                        )
                        current_weight = (
                            self.graph[path[i - 1]][path[i]]["weight"]
                            + self.graph[path[i]][path[i + 1]]["weight"]
                        )
                        if new_weight < current_weight:
                            path[i] = neighbor
                            improved = True
                            break
            if improved:
                break
        return path
    
    def grasp(self, start, end, k, max_iterations=100):
        best_paths = []
        for _ in range(max_iterations):
            path = self.greedy_randomized_construction(start, end)
            if path:
                path = self.local_search(path)
                # Dodajemy ścieżkę, jeśli nie jest identyczna z już znalezionymi
                if path not in [p[1] for p in best_paths]:
                    heapq.heappush(best_paths, (self.path_cost(path), path))
                    if len(best_paths) > k:
                        heapq.heappop(best_paths)

        best_paths = sorted(best_paths, key=lambda x: x[0])

        if len(best_paths) < k:
            print(f"Ostrzeżenie: W algorytmie GRASP znaleziono tylko {len(best_paths)} różnych ścieżek, mniej niż żądana liczba {k} ścieżek.")
        
        return best_paths
    
    def path_cost(self, path):
        return sum(
            self.graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )


if __name__ == "__main__":
    graph = GRASPGraph()

    graph.add_edge(1, 2, 2)
    graph.add_edge(1, 3, 3)
    graph.add_edge(2, 4, 1)
    graph.add_edge(2, 5, 4)
    graph.add_edge(3, 5, 2)
    graph.add_edge(3, 6, 3)
    graph.add_edge(4, 7, 3)
    graph.add_edge(5, 7, 2)
    graph.add_edge(5, 8, 1)
    graph.add_edge(6, 8, 2)
    graph.add_edge(7, 9, 1)
    graph.add_edge(7, 10, 4)
    graph.add_edge(8, 9, 3)
    graph.add_edge(9, 10, 2)

    start_node = 1
    end_node = 10
    k_paths = 3

    best_paths = graph.grasp(start_node, end_node, k_paths)
    print("Best paths:", best_paths)
