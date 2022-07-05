import random
from typing import List

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, sort_edge_index

MAX_NODES = 20
NUM_FEATURES = 1
NUM_CLASSES = 2


def _get_cycle_graph(n_nodes: int) -> Data:
    g = from_networkx(nx.cycle_graph(n_nodes))
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def _get_wheel_graph(n_nodes: int) -> Data:
    g = from_networkx(nx.wheel_graph(n_nodes))
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def gen_data() -> List[Data]:
    data = []
    for n_nodes in range(5, MAX_NODES + 1):
        g = _get_wheel_graph(n_nodes=n_nodes + 1)
        g.x = torch.ones((n_nodes + 1, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        data.append(g)

        g = _get_cycle_graph(n_nodes=n_nodes)
        g.x = torch.ones((n_nodes, 1), dtype=torch.float)
        g.y = torch.LongTensor([0])
        data.append(g)
    random.shuffle(data)
    return data


def gen_val_data() -> List[Data]:
    data = []
    for n_nodes in range(5, 8):
        g = _get_wheel_graph(n_nodes=n_nodes + 1)
        g.x = torch.ones((n_nodes + 1, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        data.append(g)
    return data


def gen_exp_data() -> List[Data]:
    data = []
    for n_nodes in range(7, MAX_NODES + 1):
        g = _get_wheel_graph(n_nodes=n_nodes + 1)
        g.x = torch.ones((n_nodes + 1, 1), dtype=torch.float)
        g.y = torch.LongTensor([1])
        data.append(g)
    return data
