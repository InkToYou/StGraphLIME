from abc import ABCMeta
from typing import List, Tuple

import numpy as np
from torch import LongTensor, Tensor
from torch_geometric.data import Data

from ..models import BaseGNNModel


class BasePerturbator(metaclass=ABCMeta):
    n_pb_data: int
    n_pb: int

    def __init__(self, n_pb_data: int, n_pb: int) -> None:
        self.n_pb_data = n_pb_data
        self.n_pb = n_pb

    def perturb(self, G: Data, model: BaseGNNModel) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def perturb_series(
        self, xs: List[Tensor], edge_indices: List[LongTensor], model: BaseGNNModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
