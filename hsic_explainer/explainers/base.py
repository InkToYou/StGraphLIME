from abc import ABCMeta
from typing import List, Literal

import numpy as np
from torch import LongTensor, Tensor
from torch_geometric.data import Data

from ..models import BaseGNNModel


class BaseExplainer(metaclass=ABCMeta):
    def explain_graph(
        self,
        G: Data,
        model: BaseGNNModel,
    ) -> np.ndarray:
        raise NotImplementedError

    def explain_graph_series(
        self,
        xs: List[Tensor],
        edge_indices: List[LongTensor],
        model: BaseGNNModel,
        y: LongTensor,
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def etype(self) -> Literal["node", "edge"]:
        raise NotImplementedError
