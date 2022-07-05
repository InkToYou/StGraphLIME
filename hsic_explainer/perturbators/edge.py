from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from ..models import BaseGNNModel
from ..utils import tonumpy_copy
from .base import BasePerturbator


class EdgeRemovePerturbator(BasePerturbator):
    @torch.no_grad()
    def perturb(self, G: Data, model: BaseGNNModel) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats = []
        preds = []

        for _ in range(self.n_pb_data):
            n_edges = G.num_edges // 2
            sub_edges = torch.LongTensor(
                np.random.choice(
                    np.arange(n_edges),
                    (n_edges - self.n_pb),
                    replace=False,
                )
            )
            fr, to = G.edge_index
            fr, to = fr[fr < to], to[fr < to]
            fr, to = fr[sub_edges], to[sub_edges]
            sub_edge_index = to_undirected(torch.stack([fr, to], dim=0))

            f = np.zeros(n_edges)
            f[sub_edges] = 1.0
            feats.append(f)

            p = softmax(
                model(x=G.x, edge_index=sub_edge_index, batch=G.batch), dim=-1
            ).squeeze()
            preds.append(tonumpy_copy(p))

        return np.array(feats), np.array(preds)
