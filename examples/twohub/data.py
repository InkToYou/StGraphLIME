import random
from typing import List

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, sort_edge_index

MAX_NODES = 15
NUM_FEATURES = 1
NUM_CLASSES = 2


def _get_2cycle_graph(n_nodes: int) -> Data:
    nxg = nx.cycle_graph(n_nodes)
    _nxg = nx.cycle_graph(n_nodes)
    nxg = nx.disjoint_union(nxg, _nxg)
    assert nxg.degree(n_nodes - 1) < n_nodes - 1
    assert nxg.degree(2 * n_nodes - 1) < n_nodes - 1
    nxg.add_edge(n_nodes - 1, 2 * n_nodes - 1)

    g = from_networkx(nxg)
    g.x = torch.ones((2 * n_nodes, 1), dtype=torch.float)
    g.y = torch.LongTensor([0])
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def _get_2wheel_graph(n_nodes: int) -> Data:
    nxg = nx.wheel_graph(n_nodes)
    _nxg = nx.wheel_graph(n_nodes)
    nxg = nx.disjoint_union(nxg, _nxg)
    assert nxg.degree(n_nodes - 1) < n_nodes - 1
    assert nxg.degree(2 * n_nodes - 1) < n_nodes - 1
    nxg.add_edge(n_nodes - 1, 2 * n_nodes - 1)

    g = from_networkx(nxg)
    g.x = torch.ones((2 * n_nodes, 1), dtype=torch.float)
    g.y = torch.LongTensor([1])
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def _get_cycle_wheel_graph(n_nodes: int) -> Data:
    nxg = nx.cycle_graph(n_nodes)
    _nxg = nx.wheel_graph(n_nodes + 1)
    nxg = nx.disjoint_union(nxg, _nxg)
    assert nxg.degree(n_nodes - 1) < n_nodes - 1
    assert nxg.degree(2 * n_nodes) < n_nodes
    nxg.add_edge(n_nodes - 1, 2 * n_nodes)
    g = from_networkx(nxg)
    g.x = torch.ones((2 * n_nodes + 1, 1), dtype=torch.float)
    g.y = torch.LongTensor([1])
    g.edge_index, _ = sort_edge_index(g.edge_index)
    return g


def gen_data() -> List[Data]:
    data = []
    for n_nodes in range(5, MAX_NODES + 1):
        g = _get_2wheel_graph(n_nodes=n_nodes + 1)
        data.append(g)

        g = _get_cycle_wheel_graph(n_nodes=n_nodes)
        data.append(g)

        g = _get_2cycle_graph(n_nodes=n_nodes)
        data.append(g)
    random.shuffle(data)
    return data


def gen_val_data() -> List[Data]:
    data = []
    for n_nodes in range(5, 7):
        g = _get_2wheel_graph(n_nodes=n_nodes)
        data.append(g)
    random.shuffle(data)
    return data


def gen_exp_data() -> List[Data]:
    data = []
    for n_nodes in range(7, MAX_NODES + 1):
        g = _get_2wheel_graph(n_nodes=n_nodes)
        data.append(g)
    random.shuffle(data)
    return data
