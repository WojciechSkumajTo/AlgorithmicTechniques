import heapq
import copy
import networkx as nx
import matplotlib.pyplot as plt


class YensGraph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, from_node, to_node, weight):
        self.edges.setdefault(from_node, []).append((to_node, weight))

    def dijkstra(self, start, end):
        queue, seen, mins = [(0, start, [])], set(), {start: 0}
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            if node not in seen:
                seen.add(node)
                path = path + [node]
                if node == end:
                    return (cost, path)

                for to_node, weight in self.edges.get(node, []):
                    if to_node in seen:
                        continue
                    prev = mins.get(to_node, None)
                    next_cost = cost + weight
                    if prev is None or next_cost < prev:
                        mins[to_node] = next_cost
                        heapq.heappush(queue, (next_cost, to_node, path))
        return (float("inf"), [])

    def yen(self, start, end, k=3):
        original_graph = copy.deepcopy(self.edges)
        paths = []

        (cost, path) = self.dijkstra(start, end)
        if cost < float("inf") and path:
            paths.append((cost, path))

        while len(paths) < k:
            best_candidate = None

            for path in paths:
                for i in range(len(path[1]) - 1):
                    spur_node = path[1][i]
                    root_path = path[1][: i + 1]

                    self.edges = copy.deepcopy(original_graph)
                    for p in paths:
                        if p[1][: i + 1] == root_path and p[1][i] in self.edges:
                            self.remove_edge(p[1][i], p[1][i + 1])

                    (spur_cost, spur_path) = self.dijkstra(spur_node, end)
                    if spur_path and spur_path[-1] == end and spur_cost < float("inf"):
                        candidate_path = root_path[:-1] + spur_path
                        candidate_cost = sum(
                            [
                                self.get_edge_cost(
                                    candidate_path[j], candidate_path[j + 1]
                                )
                                for j in range(len(candidate_path) - 1)
                            ]
                        )
                        if candidate_path not in [p[1] for p in paths] and (
                            best_candidate is None or candidate_cost < best_candidate[0]
                        ):
                            best_candidate = (candidate_cost, candidate_path)

                    self.edges = copy.deepcopy(original_graph)

            if best_candidate:
                paths.append(best_candidate)
            else:
                break

        if len(paths) < k:
            print(
                f"Ostrzeżenie: W algorytmie Yen's znaleziono tylko {len(paths)} różnych ścieżek, mniej niż żądana liczba {k} ścieżek."
            )

        return paths

    def remove_edge(self, u, v):
        self.edges[u] = [edge for edge in self.edges[u] if edge[0] != v]

    def remove_node(self, node):
        if node in self.edges:
            del self.edges[node]
        for _, edges in self.edges.items():
            edges[:] = [edge for edge in edges if edge[0] != node]

    def get_edge_cost(self, u, v):
        for to_node, weight in self.edges.get(u, []):
            if to_node == v:
                return weight
        return float("inf")


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

    START_NODE = 1
    END_NODE = 5
    K_PATHS = 100
    FILE_PATH = "/home/wojciechskumajto/AlgorithmicTechniques/example_edges.txt"

    graph = YensGraph()

    # edges = [
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (1, 4, 1),
    #     (1, 5, 1),
    #     (1, 6, 1),
    #     (1, 7, 1),
    #     (2, 3, 1),
    #     (2, 4, 1),
    #     (2, 5, 1),
    #     (2, 6, 1),
    #     (2, 7, 1),
    #     (3, 4, 1),
    #     (3, 5, 1),
    #     (3, 6, 1),
    #     (3, 7, 1),
    #     (4, 5, 1),
    #     (4, 6, 1),
    #     (4, 7, 1),
    #     (5, 6, 1),
    #     (5, 7, 1),
    #     (6, 7, 1),
    # ]

    edges = read_edges_from_file(FILE_PATH)

    for u, v, w in edges:
        graph.add_edge(u, v, w)

    shortest_paths = graph.yen(START_NODE, END_NODE, K_PATHS)

    best_path = [shortest_paths[0][1]]

    for i, path in enumerate(shortest_paths):
        print(f"Path {i+1} ==> {path}")

    create_and_visualize_graph_with_paths(edges, best_path)

if __name__ == "__main__":
   main()