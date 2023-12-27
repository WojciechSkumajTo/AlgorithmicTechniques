import random
import heapq
import networkx as nx
import matplotlib.pyplot as plt


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
                return []
            neighbors = list(self.graph.successors(current))
            if not neighbors:
                return []
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
                if path not in [p[1] for p in best_paths]:
                    heapq.heappush(best_paths, (self.path_cost(path), path))
                    if len(best_paths) > k:
                        heapq.heappop(best_paths)

        best_paths = sorted(best_paths, key=lambda x: x[0])

        if len(best_paths) < k:
            print(
                f"Ostrzeżenie: W algorytmie GRASP znaleziono tylko {len(best_paths)} różnych ścieżek, mniej niż żądana liczba {k} ścieżek."
            )

        return best_paths

    def path_cost(self, path):
        return sum(
            self.graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
        )


def read_edges_from_file(file_path):
    edges = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                u, v, w = parts
                edges.append((int(u), int(v), round(float(w), 2)))
            else:
                raise ValueError(
                    "Each line must have exactly three values separated by spaces."
                )

    return edges


def create_and_visualize_graph_with_paths(edges, paths_to_draw):
    graph_dict = {}
    for u, v, w in edges:
        if u not in graph_dict:
            graph_dict[u] = []
        int_weight = int(round(w))
        graph_dict[u].append((v, int_weight))

    G = nx.DiGraph()

    for node, edges in graph_dict.items():
        G.add_node(node)
        for edge in edges:
            G.add_edge(node, edge[0], weight=edge[1])

    pos = nx.spring_layout(G, k=0.5, iterations=20)

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", alpha=0.9)
    nx.draw_networkx_labels(
        G, pos, font_size=12, font_family="sans-serif", font_weight="bold"
    )
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1
    )

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="green", font_size=10
    )

    for path in paths_to_draw:
        edges_in_path = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges_in_path,
            edge_color="red",
            width=2.0,
            alpha=0.7,
            arrowstyle="->",
            arrowsize=20,
        )

    plt.axis("off")
    plt.tight_layout()

    plt.show()


def main(): 
    graph = GRASPGraph()

    FILE_PATH = "/home/wojciechskumajto/AlgorithmicTechniques/example_edges.txt"
    START_NODE = 1
    END_NODE = 7
    K_PATHS = 4

    edges = [
        (1, 2, 1),
        (1, 3, 1),
        (1, 4, 1),
        (1, 5, 1),
        (1, 6, 1),
        (1, 7, 1),
        (2, 3, 1),
        (2, 4, 1),
        (2, 5, 1),
        (2, 6, 1),
        (2, 7, 1),
        (3, 4, 1),
        (3, 5, 1),
        (3, 6, 1),
        (3, 7, 1),
        (4, 5, 1),
        (4, 6, 1),
        (4, 7, 1),
        (5, 6, 1),
        (5, 7, 1),
        (6, 7, 1),
    ]

    #! edges = read_edges_from_file(FILE_PATH)

    for u, v, w in edges:
        graph.add_edge(u, v, w)

    shortest_paths = graph.grasp(START_NODE, END_NODE, K_PATHS)
    
    best_path = [shortest_paths[0][1]]

    for i, path in enumerate(shortest_paths):
        print(f"Path {i+1} ==> {path}")

    create_and_visualize_graph_with_paths(edges, best_path)

if __name__ == "__main__":
    main()
