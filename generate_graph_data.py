import random


class GraphDataGenerator:
    def generate_data(self, num_nodes, additional_edges_per_node, weight_range):
        edges = []
        for i in range(1, num_nodes):
            for j in range(
                i + 1, min(i + 1 + additional_edges_per_node, num_nodes + 1)
            ):
                weight = random.randint(*weight_range)
                edges.append((i, j, weight))
        return edges

    def save_data_to_file(slef, file_path, edges):
        with open(file_path, "w") as file:
            for u, v, w in edges:
                file.write(f"{u} {v} {w}\n")

    def read_edges_from_file(self, file_path):
        edges = []

        with open(file_path, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) == 3:
                    u, v, w = parts
                    edges.append(
                        (int(u), int(v), round(float(w), 2))
                    ) 
                else:
                    raise ValueError(
                        "Each line must have exactly three values separated by spaces."
                    )

            return edges


def main():

    NUM_NODES_LIST = 50
    ADDITIONAL_EDGES_PER_NODE = 2
    WEIGHT_RANGE = (1, 15)
    FILE_PATH = '/home/wojciechskumajto/AlgorithmicTechniques/example_edges.txt'
    generator = GraphDataGenerator()
    edges = generator.generate_data(NUM_NODES_LIST, ADDITIONAL_EDGES_PER_NODE, WEIGHT_RANGE)
    generator.save_data_to_file(FILE_PATH, edges)


if __name__ == '__main__':
    main()