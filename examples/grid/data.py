from typing import List

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, sort_edge_index

SIZE_GRID = 4
NUM_FEATURES = 1
NUM_CLASSES = 4


def gen_data() -> List[Data]:
    data = []
    for nc in range(0, SIZE_GRID * SIZE_GRID):
        if nc % SIZE_GRID == SIZE_GRID - 1:
            continue
        if nc // SIZE_GRID == SIZE_GRID - 1:
            continue
        print(nc)
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        g.x[[nc, nc + 1, nc + SIZE_GRID, nc + SIZE_GRID + 1]] = 1.0
        data.append(g)
    print(f"Rectangle: {len(data)}")

    for loc in range(SIZE_GRID):
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        g.x[[loc + SIZE_GRID * i for i in range(SIZE_GRID)]] = 1.0
        data.append(g)

        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        loc = loc * SIZE_GRID
        g.x[[loc + 1 * i for i in range(SIZE_GRID)]] = 1.0
        data.append(g)
    print(f"Rec + Line: {len(data)}")

    for _ in range(10):
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([2])
        data.append(g)

    for _ in range(10):
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([3])
        is_v = np.random.choice([True, False])
        if is_v:
            loc = np.random.randint(0, SIZE_GRID)
            g.x[[loc + SIZE_GRID * i for i in range(SIZE_GRID)]] = 1.0
        else:
            loc = np.random.randint(0, SIZE_GRID)
            loc = loc * SIZE_GRID
            g.x[[loc + 1 * i for i in range(SIZE_GRID)]] = 1.0
        nc = np.random.randint(0, g.num_nodes)
        while nc % SIZE_GRID >= SIZE_GRID - 1 or nc // SIZE_GRID >= SIZE_GRID - 1:
            nc = np.random.randint(0, g.num_nodes)
        g.x[[nc, nc + 1, nc + SIZE_GRID, nc + SIZE_GRID + 1]] = 1.0
        data.append(g)

    return data


def gen_val_data() -> List:
    data = []
    for nc in range(0, 2):
        if nc % SIZE_GRID == SIZE_GRID - 1:
            continue
        if nc // SIZE_GRID == SIZE_GRID - 1:
            continue
        print(nc)
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        g.x[[nc, nc + 1, nc + SIZE_GRID, nc + SIZE_GRID + 1]] = 1.0
        data.append(g)
    print(f"Rectangle: {len(data)}")

    for loc in range(0, 1):
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        g.x[[loc + SIZE_GRID * i for i in range(SIZE_GRID)]] = 1.0
        data.append(g)

        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        loc = loc * SIZE_GRID
        g.x[[loc + 1 * i for i in range(SIZE_GRID)]] = 1.0
        data.append(g)
    print(f"Rec + Line: {len(data)}")

    return data


def gen_exp_data() -> List[Data]:
    data = []
    for nc in range(2, SIZE_GRID * SIZE_GRID):
        if nc % SIZE_GRID == SIZE_GRID - 1:
            continue
        if nc // SIZE_GRID == SIZE_GRID - 1:
            continue
        print(nc)
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        g.x[[nc, nc + 1, nc + SIZE_GRID, nc + SIZE_GRID + 1]] = 1.0
        data.append(g)
    print(f"Rectangle: {len(data)}")

    for loc in range(1, SIZE_GRID):
        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        g.x[[loc + SIZE_GRID * i for i in range(SIZE_GRID)]] = 1.0
        data.append(g)

        grid = nx.grid_graph((SIZE_GRID, SIZE_GRID))
        grid = nx.convert_node_labels_to_integers(grid)
        g = from_networkx(grid)
        g.edge_index, _ = sort_edge_index(g.edge_index)
        g.x = torch.zeros((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        loc = loc * SIZE_GRID
        g.x[[loc + 1 * i for i in range(SIZE_GRID)]] = 1.0
        data.append(g)
    print(f"Rec + Line: {len(data)}")

    return data
