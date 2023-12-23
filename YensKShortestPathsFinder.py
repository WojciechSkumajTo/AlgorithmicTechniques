import heapq
import copy
import networkx as nx

class Graph:
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

    def yen(self, start, end, k):
        original_graph = copy.deepcopy(self.edges)
        paths = []
        (cost, path) = self.dijkstra(start, end)
        if cost < float("inf") and path:
            paths.append((cost, path))

        for _ in range(1, k):
            last_path = paths[-1][1]
            for i in range(len(last_path) - 1):
                spur_node = last_path[i]
                root_path = last_path[:i + 1]

                self.edges = copy.deepcopy(original_graph)

                # Remove nodes and edges
                for p in paths:
                    if len(p[1]) > i and root_path == p[1][:i + 1]:
                        self.remove_edge(p[1][i], p[1][i + 1])
                        if p[1][i + 1] != spur_node:
                            self.remove_node(p[1][i + 1])

                (spur_cost, spur_path) = self.dijkstra(spur_node, end)
                if spur_path and spur_path[-1] == end and spur_path not in [p[1] for p in paths]:
                    total_cost = sum([self.get_edge_cost(root_path[j], root_path[j + 1]) for j in range(len(root_path) - 1)]) + spur_cost
                    candidate_path = root_path[:-1] + spur_path
                    if candidate_path not in [p[1] for p in paths]:
                        paths.append((total_cost, candidate_path))

                self.edges = copy.deepcopy(original_graph)

            if not paths or len(paths) <= 1:
                break
            paths.sort(key=lambda x: x[0])

        self.edges = original_graph
        return paths

    def remove_edge(self, u, v):
        self.edges[u] = [edge for edge in self.edges[u] if edge[0] != v]

    def remove_node(self, node):
        if node in self.edges:
            del self.edges[node]
        for _, edges in self.edges.items():
            edges[:] = [edge for edge in edges if edge[0] != node]

    def get_edge_cost(self, u, v):
        for (to_node, weight) in self.edges.get(u, []):
            if to_node == v:
                return weight
        return float('inf')

# Example Usage
if __name__ == "__main__":
    graph = Graph()

    graph.add_edge(1, 2, 2)
    graph.add_edge(1, 3, 4)
    graph.add_edge(2, 4, 3)
    graph.add_edge(3, 4, 2)
    graph.add_edge(4, 5, 1)
    graph.add_edge(4, 6, 3)
    graph.add_edge(5, 7, 2)
    graph.add_edge(6, 7, 1)

    start_node = 1
    end_node = 7
    k_paths = 3

    shortest_paths = graph.yen(start_node, end_node, k_paths)
    print("Shortest paths:", shortest_paths)
