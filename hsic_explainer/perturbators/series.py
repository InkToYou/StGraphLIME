import copy
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax

from ..models import BaseTGNNModel
from ..utils import tonumpy_copy
from .base import BasePerturbator
from .node import get_subset_of_nodes_with_random_walk


class RandomWalkNodeFeatureSeriesPerturbator(BasePerturbator):
    @torch.no_grad()
    def perturb_series(
        self,
        xs: List[torch.Tensor],
        edge_indices: List[torch.LongTensor],
        model: BaseTGNNModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        feats, preds = [], []

        for _ in range(self.n_pb_data):
            at = np.random.randint(0, len(xs))
            pb_xs = copy.deepcopy(xs)
            f = np.array([])
            for t in range(len(xs)):
                if t == at:
                    sub_nodes = get_subset_of_nodes_with_random_walk(
                        edge_index=edge_indices[t], walk_size=self.n_pb
                    )
                    for v in sub_nodes:
                        if pb_xs[t][v].item() < 0.5:
                            pb_xs[t][v] = pb_xs[t][v] + np.random.rand()
                        else:
                            pb_xs[t][v] = pb_xs[t][v] - np.random.rand()

                f = np.append(f, tonumpy_copy(pb_xs[t]))

            feats.append(f)
            p = softmax(
                model(xs=pb_xs, edge_indices=edge_indices),
                dim=-1,
            ).squeeze()
            preds.append(tonumpy_copy(p))

        # Add the original prediction
        ori_p = softmax(
            model(xs=xs, edge_indices=edge_indices),
            dim=-1,
        ).squeeze()
        preds.append(tonumpy_copy(ori_p))

        ori_xs = np.array([])
        for t in range(len(xs)):
            ori_xs = np.append(ori_xs, tonumpy_copy(xs[t]))
        feats.append(ori_xs)

        return np.array(feats), np.array(preds)
