import os
import random
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx


def fix_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def tonumpy_copy(x: torch.Tensor) -> np.ndarray:
    x_np: np.ndarray = x.detach().cpu().numpy().copy()
    return x_np


def extract_subG(
    sub_nodes: np.ndarray, edge_index: torch.LongTensor
) -> Tuple[np.ndarray, torch.LongTensor]:
    sub_edge_index, _ = subgraph(
        torch.LongTensor(sub_nodes),
        edge_index,
        relabel_nodes=False,
        num_nodes=None,
    )
    n_nodes = int(edge_index.max().item() + 1)
    n_idx = torch.zeros(n_nodes).long()
    n_idx[sub_nodes.tolist()] = torch.arange(sub_nodes.shape[0])
    sub_edge_index = torch.LongTensor(n_idx[sub_edge_index])

    return sub_nodes, sub_edge_index


def edge_score2node_score(G: Data, edge_score: np.ndarray) -> np.ndarray:
    if edge_score.shape == (G.edge_index.shape[1],):
        # For edge_mask
        data = Data(
            edge_index=G.edge_index, edge_score=edge_score, num_nodes=G.num_nodes
        )
        nxg = to_networkx(data, edge_attrs=["edge_score"])
        node_score = np.zeros(data.num_nodes)
        for s, t, d in nxg.edges(data=True):
            node_score[s] = max(node_score[s], d["edge_score"])
            node_score[t] = max(node_score[t], d["edge_score"])
    elif edge_score.shape == (G.edge_index.shape[1] // 2,):
        # For HSICExplainer
        node_score = np.zeros(G.num_nodes)
        for i, (s, t) in zip(edge_score, G.edge_index.T.numpy()):
            node_score[s] = max(node_score[s], i)
            node_score[t] = max(node_score[t], i)
    else:
        print(edge_score.shape, G.edge_index.shape)
        raise NotImplementedError

    return node_score


def get_subset_of_nodes_with_random_walk(
    edge_index: torch.LongTensor, walk_size: int
) -> np.ndarray:
    n = int(edge_index.max().item()) + 1

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


def draw_graph(
    g: nx.Graph,
    ax,
    title: str,
    labels: Optional[dict] = None,
    with_labels: bool = False,
    node_color: Union[np.ndarray, str] = "#A0CBE2",
    edge_color: Union[np.ndarray, str] = "#A0CBE2",
) -> None:
    pos = nx.drawing.layout.spring_layout(g, seed=42, iterations=50)
    cmap = plt.cm.plasma
    vmin, vmax = 0.0, 0.0
    if not isinstance(node_color, str):
        vmax = max(max(node_color), vmax)
    if not isinstance(edge_color, str):
        vmax = max(max(edge_color), vmax)

    nx.draw(
        g,
        pos=pos,
        ax=ax,
        arrows=False,
        labels=labels,
        alpha=0.5,
        with_labels=with_labels,
        node_color=node_color,
        edge_color=edge_color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
    )

    ax.set_title(title)
