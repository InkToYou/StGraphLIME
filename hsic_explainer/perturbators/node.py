from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from torch_geometric.data import Data

from ..models import BaseGNNModel
from ..utils import extract_subG, get_subset_of_nodes_with_random_walk, tonumpy_copy
from .base import BasePerturbator


class NodeRemovePerturbator(BasePerturbator):
    @torch.no_grad()
    def perturb(self, G: Data, model: BaseGNNModel) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats, preds = [], []

        n_nodes = G.num_nodes
        for _ in range(self.n_pb_data):
            sub_nodes = np.random.choice(
                np.arange(n_nodes),
                (n_nodes - self.n_pb),
                replace=False,
            )
            sub_nodes = np.sort(sub_nodes)
            sub_nodes, sub_edge_index = extract_subG(
                sub_nodes=sub_nodes, edge_index=G.edge_index
            )

            f = np.zeros(n_nodes)
            f[sub_nodes] = 1.0
            feats.append(f)

            p = softmax(
                model(
                    x=G.x[sub_nodes],
                    edge_index=sub_edge_index,
                    batch=G.batch[sub_nodes],
                ),
                dim=-1,
            ).squeeze()
            preds.append(tonumpy_copy(p))

        # Add the original prediction
        ori_p = softmax(
            model(x=G.x, edge_index=G.edge_index, batch=G.batch),
            dim=-1,
        ).squeeze()
        preds.append(tonumpy_copy(ori_p))
        feats.append(np.ones(n_nodes))

        return np.array(feats), np.array(preds)


class RandomWalkNodeRemovePerturbator(BasePerturbator):
    @torch.no_grad()
    def perturb(self, G: Data, model: BaseGNNModel) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats, preds = [], []

        n_nodes = G.num_nodes
        for _ in range(self.n_pb_data):
            rm_nodes = get_subset_of_nodes_with_random_walk(
                edge_index=G.edge_index, walk_size=self.n_pb
            )
            sub_nodes = np.setdiff1d(np.arange(n_nodes), rm_nodes)
            sub_nodes = np.sort(sub_nodes)
            sub_nodes, sub_edge_index = extract_subG(
                sub_nodes=sub_nodes, edge_index=G.edge_index
            )

            f = np.zeros(n_nodes)
            f[sub_nodes] = 1.0
            feats.append(f)

            p = softmax(
                model(
                    x=G.x[sub_nodes],
                    edge_index=sub_edge_index,
                    batch=G.batch[sub_nodes],
                ),
                dim=-1,
            ).squeeze()
            preds.append(tonumpy_copy(p))

        # Add the original prediction
        ori_p = softmax(
            model(x=G.x, edge_index=G.edge_index, batch=G.batch),
            dim=-1,
        ).squeeze()
        preds.append(tonumpy_copy(ori_p))
        feats.append(np.ones(n_nodes))

        return np.array(feats), np.array(preds)


class NodeFeatureNoisePerturbator(BasePerturbator):
    @torch.no_grad()
    def perturb(self, G: Data, model: BaseGNNModel) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats, preds = [], []

        n_nodes = G.num_nodes
        for _ in range(self.n_pb_data):
            sub_nodes = np.random.choice(
                np.arange(n_nodes),
                (n_nodes - self.n_pb),
                replace=False,
            )
            pb_vec = torch.ones(G.x.shape)
            for v in sub_nodes:
                if G.x[v] < 0.5:
                    pb_vec[v] = 0.5 * np.random.rand()
                else:
                    pb_vec[v] = -0.5 * np.random.rand()

            f = (G.x * pb_vec).numpy().squeeze()
            feats.append(f)

            p = softmax(
                model(x=G.x + pb_vec, edge_index=G.edge_index, batch=G.batch), dim=-1
            ).squeeze()
            preds.append(tonumpy_copy(p))

        # Add the original prediction
        ori_p = softmax(
            model(x=G.x, edge_index=G.edge_index, batch=G.batch),
            dim=-1,
        ).squeeze()
        preds.append(tonumpy_copy(ori_p))
        feats.append(tonumpy_copy(G.x.squeeze()))

        return np.array(feats), np.array(preds)


class RandomWalkNodeFeatureZeroPerturbator(BasePerturbator):
    @torch.no_grad()
    def perturb(self, G: Data, model: BaseGNNModel) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats, preds = [], []

        for _ in range(self.n_pb_data):
            sub_nodes = get_subset_of_nodes_with_random_walk(
                edge_index=G.edge_index, walk_size=self.n_pb
            )

            pb_vec = torch.ones(G.x.shape)
            pb_vec[sub_nodes.tolist()] = 0.0

            f = G.x.argmax(dim=1).numpy()
            f[sub_nodes.tolist()] = -1  # Set an unused value
            feats.append(f)

            p = softmax(
                model(x=G.x * pb_vec, edge_index=G.edge_index, batch=G.batch), dim=-1
            ).squeeze()
            preds.append(tonumpy_copy(p))

        # Add the original prediction
        ori_p = softmax(
            model(x=G.x, edge_index=G.edge_index, batch=G.batch),
            dim=-1,
        ).squeeze()
        preds.append(tonumpy_copy(ori_p))
        feats.append(tonumpy_copy(G.x.argmax(dim=1).squeeze()))

        return np.array(feats), np.array(preds)
