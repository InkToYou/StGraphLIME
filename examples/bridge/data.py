import random
from typing import List

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, sort_edge_index

MAX_NODES = 10
NUM_FEATURES = 1
NUM_CLASSES = 2


def _get_two_cycles_graph(n_nodes: int) -> Data:
    g_1 = nx.cycle_graph(n_nodes)
    g_2 = nx.cycle_graph(n_nodes)
    g = nx.disjoint_union(g_1, g_2)
    g = from_networkx(g)
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def _get_glasses_graph(n_nodes: int) -> Data:
    g_1 = nx.cycle_graph(n_nodes)
    g_2 = nx.cycle_graph(n_nodes)
    g = nx.disjoint_union(g_1, g_2)
    g.add_edge(0, n_nodes)
    g = from_networkx(g)
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def gen_data() -> List[Data]:
    data = []
    for n_nodes in range(3, MAX_NODES + 1):
        g = _get_glasses_graph(n_nodes=n_nodes)
        g.x = torch.ones((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        data.append(g)

        g = _get_two_cycles_graph(n_nodes=n_nodes)
        g.x = torch.ones((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        data.append(g)

    random.shuffle(data)
    return data


def gen_val_data() -> List[Data]:
    data = []
    for n_nodes in range(3, 5):
        g = _get_glasses_graph(n_nodes=n_nodes)
        g.x = torch.ones((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        data.append(g)

    return data


def gen_exp_data() -> List[Data]:
    data = []
    for n_nodes in range(5, MAX_NODES + 1):
        g = _get_glasses_graph(n_nodes=n_nodes)
        g.x = torch.ones((g.num_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        data.append(g)

    return data
