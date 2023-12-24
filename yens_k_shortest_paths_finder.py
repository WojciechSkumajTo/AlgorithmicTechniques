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

        # Pierwsze wyszukiwanie najkrótszej ścieżki
        (cost, path) = self.dijkstra(start, end)
        if cost < float("inf") and path:
            paths.append((cost, path))

        while len(paths) < k:
            best_candidate = None

            for path in paths:
                for i in range(len(path[1]) - 1):
                    spur_node = path[1][i]
                    root_path = path[1][:i + 1]

                    self.edges = copy.deepcopy(original_graph)
                    for p in paths:
                        if p[1][:i + 1] == root_path and p[1][i] in self.edges:
                            self.remove_edge(p[1][i], p[1][i + 1])

                    (spur_cost, spur_path) = self.dijkstra(spur_node, end)
                    if spur_path and spur_path[-1] == end and spur_cost < float("inf"):
                        candidate_path = root_path[:-1] + spur_path
                        candidate_cost = sum([self.get_edge_cost(candidate_path[j], candidate_path[j + 1]) for j in range(len(candidate_path) - 1)])
                        if candidate_path not in [p[1] for p in paths] and (best_candidate is None or candidate_cost < best_candidate[0]):
                            best_candidate = (candidate_cost, candidate_path)

                    self.edges = copy.deepcopy(original_graph)

            if best_candidate:
                paths.append(best_candidate)
            else:
                break

        if len(paths) < k:
            print(f"Ostrzeżenie: W algorytmie Yen's znaleziono tylko {len(paths)} różnych ścieżek, mniej niż żądana liczba {k} ścieżek.")

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


def visualize_graph(graph):
    G = nx.DiGraph()
    for node, edges in graph.edges.items():
        G.add_node(node)
        for edge in edges:
            G.add_edge(node, edge[0], weight=edge[1])

    # Choose a layout that spreads out the nodes and makes the graph easier to read
    pos = nx.kamada_kawai_layout(G)

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="lightblue",
        font_size=15,
        font_weight="bold",
        edge_color="gray",
    )

    # Draw edge labels
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    # Show plot
    plt.show()


if __name__ == "__main__":
    graph = YensGraph()

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
    

    # visualize_graph(graph)
    
    start_node = 1
    end_node = 10
    k_paths = 3

    
    shortest_paths = graph.yen(start_node, end_node, k_paths)
    print("Shortest paths:", shortest_paths)
