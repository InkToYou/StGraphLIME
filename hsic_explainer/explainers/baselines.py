import copy
from typing import List, Literal

import numpy as np
import torch
from dig.xgraph.method import PGExplainer as _PGExplainer
from dig.xgraph.method import SubgraphX as _SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer as _GNNExplainer

from ..models import BaseGNNModel
from ..utils import tonumpy_copy
from .base import BaseExplainer

__old_loss__ = _GNNExplainer.__loss__


def __new_loss__(self, node_idx, log_logits, pred_label):
    loss = __old_loss__(self, node_idx, log_logits, pred_label)
    print(loss.item())
    return loss


_GNNExplainer.__loss__ = __new_loss__


class GNNExplainer(BaseExplainer):
    def explain_graph(
        self,
        G: Data,
        model: BaseGNNModel,
    ) -> np.ndarray:
        model.train()

        explainer = _GNNExplainer(
            model=model, epochs=1000, lr=0.005, return_type="raw", log=True
        )
        _, edge_mask = explainer.explain_graph(
            x=G.x,
            edge_index=G.edge_index,
        )
        res: np.ndarray = tonumpy_copy(edge_mask)
        return res

    @property
    def etype(self) -> Literal["node", "edge"]:
        return "edge"


class PGExplainer(BaseExplainer):
    def explain_graph(
        self,
        G: Data,
        model: BaseGNNModel,
    ) -> np.ndarray:

        with torch.no_grad():
            model.eval()
            _ = model(x=G.x, edge_index=G.edge_index, batch=G.batch)
            embed = model.get_node_embeddings()

        model.train()
        explainer = _PGExplainer(
            model=model,
            device="cpu",
            in_channels=embed.shape[1] * 2,
            explain_graph=True,
            epochs=1000,
            lr=0.005,
        )
        _, edge_mask = explainer.explain(x=G.x, edge_index=G.edge_index, embed=embed)
        res: np.ndarray = tonumpy_copy(edge_mask)
        return res

    @property
    def etype(self) -> Literal["node", "edge"]:
        return "edge"


class SubgraphX(BaseExplainer):
    def __init__(
        self,
        max_size: float,
        subgraph_building_method: Literal["split", "zero_filling"],
    ) -> None:
        self.max_size = max_size
        self.subgraph_building_method = subgraph_building_method

    @torch.no_grad()
    def explain_graph(
        self,
        G: Data,
        model: BaseGNNModel,
    ) -> np.ndarray:
        model.eval()
        pred = softmax(
            model(x=G.x, edge_index=G.edge_index, batch=G.batch), dim=-1
        ).squeeze()

        _pred = softmax(
            model(x=G.x, edge_index=G.edge_index, batch=G.batch), dim=-1
        ).squeeze()
        assert torch.equal(pred, _pred), f"{pred}, {_pred}"

        if self.max_size < 1:
            max_nodes = max(1, int(G.num_nodes * self.max_size))
        else:
            max_nodes = int(self.max_size)

        explainer = _SubgraphX(
            model=model,
            reward_method="mc_l_shapley",
            min_atoms=max_nodes,
            subgraph_building_method=self.subgraph_building_method,
            explain_graph=True,
            device="cpu",
            num_classes=pred.shape[0],
        )
        _, explanation_results, _ = explainer(
            x=G.x,
            edge_index=G.edge_index,
            max_nodes=max_nodes,
        )

        results = explanation_results[pred.argmax().item()]
        tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)
        nodelist = list(tree_node_x.coalition)

        scores = np.array([1.0 if v in nodelist else 0.0 for v in range(G.num_nodes)])
        return scores

    @property
    def etype(self) -> Literal["node", "edge"]:
        return "node"

    @property
    def name(self) -> str:
        return self.__class__.__name__ + str(self.max_size)


class RandomNodeExplainer(BaseExplainer):
    @torch.no_grad()
    def explain_graph(
        self,
        G: Data,
        model: BaseGNNModel,
    ) -> np.ndarray:

        return np.random.rand(G.num_nodes)

    @torch.no_grad()
    def explain_graph_series(
        self,
        xs: List[torch.Tensor],
        edge_indices: List[torch.LongTensor],
        model: BaseGNNModel,
        y: torch.LongTensor,
    ) -> np.ndarray:

        node_score: np.ndarray = np.random.rand(len(xs) * xs[0].shape[0])
        return node_score

    @property
    def etype(self) -> Literal["node", "edge"]:
        return "node"


class OcclusionNodeFeatureExplainer(BaseExplainer):
    @torch.no_grad()
    def explain_graph_series(
        self,
        xs: List[torch.Tensor],
        edge_indices: List[torch.LongTensor],
        model: BaseGNNModel,
        y: torch.LongTensor,
    ) -> np.ndarray:
        model.eval()

        scores = np.array([])
        ori_p = (
            softmax(model(xs=xs, edge_indices=edge_indices), dim=-1)
            .squeeze()[int(y.item())]
            .item()
        )
        for t in range(len(xs)):
            for i in range(xs[t].shape[0]):
                pb_xs = copy.deepcopy(xs)
                pb_xs[t][i] = 0.0
                pb_p = (
                    softmax(
                        model(
                            xs=pb_xs,
                            edge_indices=edge_indices,
                        ),
                        dim=-1,
                    )
                    .squeeze()[int(y.item())]
                    .item()
                )
                scores = np.append(scores, max(ori_p - pb_p, 0))

        return scores

    @property
    def etype(self) -> Literal["node", "edge"]:
        return "node"
