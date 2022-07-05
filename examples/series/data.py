import random
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx, sort_edge_index

NUM_FEATURES = 1
NUM_CLASSES = 2
N_NODES = 20
N_SEQ = 3

tree = nx.barabasi_albert_graph(N_NODES, 2)


def _get_subset_of_nodes_with_random_walk(
    edge_index: torch.LongTensor, walk_size: int
) -> np.ndarray:
    n = int(edge_index.max().item()) + 1
    assert (
        n > walk_size
    ), f"`walk_size` {walk_size} must be smaller than the number of edges {n}."

    edge_list: Dict[int, List[int]] = {i: [] for i in range(n)}
    for fr, to in zip(edge_index[0], edge_index[1]):
        edge_list[fr.item()].append(to.item())

    st = np.random.randint(n)
    sub_nodes = [st]
    cur = st
    for _ in range(walk_size):
        cur = np.random.choice(edge_list[cur])
        sub_nodes.append(cur)

    return np.array(sub_nodes)


def gen_data() -> List[tuple]:
    data: List[tuple] = []
    for _ in range(100):
        xs = []
        edge_indices = []
        for t in range(N_SEQ):
            g = from_networkx(tree)
            g.edge_index, _ = sort_edge_index(g.edge_index)
            g.x = torch.zeros((N_NODES, 1), dtype=torch.float32)
            sub_nodes = np.random.choice(
                np.arange(N_NODES),
                2,
                replace=False,
            )
            g.x[sub_nodes.tolist()] = 1.0
            xs.append(g.x)
            edge_indices.append(g.edge_index)
        data.append((xs, edge_indices, torch.LongTensor([0])))

        xs = []
        edge_indices = []
        at = np.random.randint(0, N_SEQ)
        for t in range(N_SEQ):
            g = from_networkx(tree)
            g.edge_index, _ = sort_edge_index(g.edge_index)
            g.x = torch.zeros((N_NODES, 1), dtype=torch.float32)
            if t == at:
                sub_nodes = _get_subset_of_nodes_with_random_walk(g.edge_index, 10)
                g.x[sub_nodes.tolist()] = 1.0
            else:
                sub_nodes = np.random.choice(
                    np.arange(N_NODES),
                    2,
                    replace=False,
                )
                g.x[sub_nodes.tolist()] = 1.0
            xs.append(g.x)
            edge_indices.append(g.edge_index)
        data.append((xs, edge_indices, torch.LongTensor([1])))

    random.shuffle(data)
    return data


def gen_val_data() -> List[tuple]:
    data: List[tuple] = []
    for _ in range(5):
        xs = []
        edge_indices = []
        at = np.random.randint(0, N_SEQ)
        for t in range(N_SEQ):
            g = from_networkx(tree)
            g.edge_index, _ = sort_edge_index(g.edge_index)
            g.x = torch.zeros((N_NODES, 1), dtype=torch.float32)
            if t == at:
                sub_nodes = _get_subset_of_nodes_with_random_walk(g.edge_index, 10)
                g.x[sub_nodes.tolist()] = 1.0
            else:
                sub_nodes = np.random.choice(
                    np.arange(N_NODES),
                    2,
                    replace=False,
                )
                g.x[sub_nodes.tolist()] = 1.0
            xs.append(g.x)
            edge_indices.append(g.edge_index)
        data.append((xs, edge_indices, torch.LongTensor([1])))

    return data


def gen_exp_data() -> List[tuple]:
    data: List[tuple] = []
    for _ in range(100):
        xs = []
        edge_indices = []
        at = np.random.randint(0, N_SEQ)
        for t in range(N_SEQ):
            g = from_networkx(tree)
            g.edge_index, _ = sort_edge_index(g.edge_index)
            g.x = torch.zeros((N_NODES, 1), dtype=torch.float32)
            if t == at:
                sub_nodes = _get_subset_of_nodes_with_random_walk(g.edge_index, 10)
                g.x[sub_nodes.tolist()] = 1.0
            else:
                sub_nodes = np.random.choice(
                    np.arange(N_NODES),
                    2,
                    replace=False,
                )
                g.x[sub_nodes.tolist()] = 1.0
            xs.append(g.x)
            edge_indices.append(g.edge_index)
        data.append((xs, edge_indices, torch.LongTensor([1])))

    return data


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    trn_data = gen_data()
    val_data = gen_val_data()
    exp_data = gen_exp_data()

    trn_path = Path(__file__).parent / "trn_data.data"
    val_path = Path(__file__).parent / "val_data.data"
    exp_path = Path(__file__).parent / "exp_data.data"

    # with open(trn_path, "wb") as tf:
    #     pickle.dump(trn_data, tf)
    # with open(val_path, "wb") as vf:
    #     pickle.dump(val_data, vf)
    with open(exp_path, "wb") as ef:
        pickle.dump(exp_data, ef)
