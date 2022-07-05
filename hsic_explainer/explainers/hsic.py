from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Literal

import cvxpy as cp
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import LassoLars
from spams import fistaFlat
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from ..models import BaseGNNModel
from ..perturbators.base import BasePerturbator
from ..utils import extract_subG
from .base import BaseExplainer

EPS = 1.0e-09


def _centering(A: np.ndarray) -> np.ndarray:
    HA = A - np.mean(A, axis=0, keepdims=True)
    HAH: np.ndarray = HA - np.mean(HA, axis=1, keepdims=True)

    return HAH


def _normalize(A: np.ndarray) -> np.ndarray:
    nA: np.ndarray = A / (
        np.linalg.norm(A, ord="fro", axis=(0, 1), keepdims=True) + EPS
    )
    return nA


def _compute_gram_matrix_gauss_feat(X: np.ndarray, std=3.0) -> np.ndarray:

    if X.ndim == 1:
        X = X[:, np.newaxis]

    n, d = X.shape  # n: samples, d: nodes or dims

    dist = X.reshape(1, n, d) - X.reshape(n, 1, d)
    dist = dist ** 2

    K: np.ndarray = np.exp(-(dist ** 2) / (2 * std ** 2 + EPS))

    # Centering（=HKH）
    K = _centering(K)

    # Normalizing
    K = _normalize(K)

    return K


def _compute_gram_matrix_delta_feat(X: np.ndarray, n_X: int) -> np.ndarray:
    if X.ndim == 1:
        X = X[:, np.newaxis]

    n, d = X.shape

    diff = X.reshape(1, n, d) - X.reshape(n, 1, d)
    K: np.ndarray = 1 - diff.astype(bool).astype(float)
    K = K / n_X

    # Centering（=HKH）
    K = _centering(K)

    # Normalizing
    K = _normalize(K)

    return K


def _compute_gram_matrix_gauss_pred(X: np.ndarray, std=3.0) -> np.ndarray:

    if X.ndim == 1:
        X = X[:, np.newaxis]

    n, d = X.shape  # n: samples, d: dims

    dist = X.reshape(1, n, d) - X.reshape(n, 1, d)
    dist = dist ** 2
    dist = dist.sum(axis=2)

    L: np.ndarray = np.exp(-(dist ** 2) / (2 * std ** 2 + EPS))

    # Centering（=HKH）
    L = _centering(L)

    # Normalizing
    L = _normalize(L)

    return L


class BaseHSICLasso(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(
        self, feats: np.ndarray, preds: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


def _compute_gram_matrix(X: np.ndarray, type: str, kernel: str) -> np.ndarray:
    K: np.ndarray = np.array([])
    if type == "feat":
        if kernel == "Delta":
            K = _compute_gram_matrix_delta_feat(X=X, n_X=np.unique(X).size)
        elif kernel == "Gauss":
            K = _compute_gram_matrix_gauss_feat(X=X)
        else:
            raise NotImplementedError
    elif type == "pred":
        if kernel == "Gauss":
            K = _compute_gram_matrix_gauss_pred(X=X)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return K


class HSICLasso(BaseHSICLasso):
    feat_kernel: str
    pred_kernel: str
    alpha: float

    def __init__(self, feat_kernel: str, pred_kernel: str, alpha: float) -> None:
        self.feat_kernel = feat_kernel
        self.pred_kernel = pred_kernel
        self.alpha = alpha

    def run(
        self, feats: np.ndarray, preds: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        fK = _compute_gram_matrix(X=feats, type="feat", kernel=self.feat_kernel)
        pK = _compute_gram_matrix(X=preds, type="pred", kernel=self.pred_kernel)

        n, d = feats.shape
        fK_vec = fK.reshape(n ** 2, d)

        pK_vec = pK.reshape(
            n ** 2,
        )

        print("feat", feats.min(), feats.max())
        print("pred", preds.min(), preds.max())
        print("fK_vec", fK_vec.min(), fK_vec.max())
        print("pK_vec", pK_vec.min(), pK_vec.max())
        solver = LassoLars(
            alpha=self.alpha, fit_intercept=False, normalize=False, positive=True
        )

        solver.fit(fK_vec, pK_vec)
        coef: np.ndarray = solver.coef_

        return coef

    @property
    def name(self) -> str:
        return self.__class__.__name__ + f"_{self.alpha}"


def _get_latent_groups_with_rw(
    edge_index: np.ndarray, walk_size: int
) -> Dict[int, List[int]]:

    n = edge_index.max() + 1
    groups: Dict[int, List[int]] = {i: [] for i in range(n)}
    edge_list: Dict[int, List[int]] = {i: [] for i in range(n)}
    for fr, to in zip(edge_index[0], edge_index[1]):
        edge_list[fr].append(to)

    for i in range(n):
        st = i
        sub_nodes = [st]
        cur = st
        for _ in range(walk_size):
            cur = np.random.choice(edge_list[cur])
            sub_nodes.append(cur)
        groups[i] = np.unique(sub_nodes).tolist()

    return groups


class HSICLatentGroupLasso(BaseHSICLasso):
    params: Dict[str, Any]
    feat_kernel: str
    pred_kernel: str
    alpha: float

    def __init__(self, feat_kernel: str, pred_kernel: str, alpha: float) -> None:
        self.feat_kernel = feat_kernel
        self.pred_kernel = pred_kernel
        self.alpha = alpha

        self.params = {
            "pos": True,
            "verbose": False,
            "numThreads": -1,
            "intercept": False,
            "lambda1": alpha,
            "regul": "group-lasso-l2",
            "loss": "square",
        }

    def run(
        self,
        feats: np.ndarray,
        preds: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        fK = _compute_gram_matrix(X=feats, type="feat", kernel=self.feat_kernel)
        pK = _compute_gram_matrix(X=preds, type="pred", kernel=self.pred_kernel)

        n, d = feats.shape
        fK_vec = fK.reshape(n ** 2, d)
        fK_vec = np.asfortranarray(fK_vec)

        groups = _get_latent_groups_with_rw(edge_index, walk_size=5)

        coef_to_group = []
        coef_to_val = []
        fK_vec_tilda = np.zeros((n ** 2, 0))
        for i, g in enumerate(groups.values()):
            fK_vec_tilda = np.concatenate([fK_vec_tilda, fK_vec[:, g]], axis=1)
            coef_to_group += [i + 1] * len(g)
            coef_to_val += g
        self.params["groups"] = np.array(coef_to_group, dtype=np.int32)
        fK_vec_tilda = np.asfortranarray(fK_vec_tilda)

        pK_vec = pK.reshape(n ** 2, 1).astype(np.float64)
        pK_vec = np.asfortranarray(pK_vec)

        w0 = np.ones((fK_vec_tilda.shape[1], 1), dtype=np.float64)

        coef_l = fistaFlat(pK_vec, fK_vec_tilda, w0, **self.params)
        coef_l = np.array(coef_l).flatten()

        coef = np.zeros(d)
        for c_l, ctv in zip(coef_l, coef_to_val):
            coef[ctv] += c_l

        return coef

    @property
    def name(self) -> str:
        return self.__class__.__name__ + f"_{self.alpha}"


class HSICFusedLasso(BaseHSICLasso):
    feat_kernel: str
    pred_kernel: str
    alpha1: float
    alpha2: float

    def __init__(
        self,
        feat_kernel: str,
        pred_kernel: str,
        alpha1: float,
        alpha2: float,
    ) -> None:
        self.feat_kernel = feat_kernel
        self.pred_kernel = pred_kernel
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def run(
        self,
        feats: np.ndarray,
        preds: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        fK = _compute_gram_matrix(X=feats, type="feat", kernel=self.feat_kernel)
        pK = _compute_gram_matrix(X=preds, type="pred", kernel=self.pred_kernel)

        n, d = feats.shape  # n: samples, d: nodes

        fK_vec = fK.reshape(n ** 2, d)
        pK_vec = pK.reshape(
            n ** 2,
        )

        fr, to = edge_index
        fr, to = fr[fr < to], to[fr < to]
        n_edges = fr.shape[0]
        data = [1] * n_edges + [-1] * n_edges
        row = np.tile(np.arange(n_edges), 2)
        col = np.hstack((fr, to))
        D = csr_matrix((data, (row, col)), shape=(n_edges, d))

        beta = cp.Variable(d)
        objective = cp.Minimize(
            cp.norm2(fK_vec @ beta - pK_vec) ** 2
            + self.alpha1 * cp.norm1(beta)
            + self.alpha2 * cp.norm1(D @ beta)
        )
        constraints = [0.0 <= beta]
        prob = cp.Problem(objective, constraints)
        _ = prob.solve(verbose=False)
        assert prob.status == cp.OPTIMAL, "Failed to solve FusedLasso."

        coef: np.ndarray = np.array(beta.value)

        return coef

    @property
    def name(self) -> str:
        return self.__class__.__name__ + f"_{self.alpha1}_{self.alpha2}"


class HSICExplainer(BaseExplainer):
    def __init__(
        self,
        hsic_lasso: BaseHSICLasso,
        perturbator: BasePerturbator,
        etype: Literal["node", "edge"],
    ) -> None:
        self.hsic_lasso = hsic_lasso
        self.perturbator = perturbator
        self._etype = etype

    @torch.no_grad()
    def explain_graph(
        self,
        G: Data,
        model: BaseGNNModel,
    ) -> np.ndarray:
        model.eval()

        feats, preds = self.perturbator.perturb(G=G, model=model)
        print(feats.shape, preds.shape)

        score = self.hsic_lasso.run(feats, preds, G.edge_index.numpy())
        return score

    @torch.no_grad()
    def explain_graph_series(
        self,
        xs: List[torch.Tensor],
        edge_indices: List[torch.LongTensor],
        model: BaseGNNModel,
        y: torch.LongTensor,
    ) -> np.ndarray:
        model.eval()

        feats, preds = self.perturbator.perturb_series(
            xs=xs, edge_indices=edge_indices, model=model
        )
        print(feats.shape, preds.shape)

        edge_index = np.zeros((2, 0))
        n_nodes = edge_indices[0].max().item() + 1
        for i, e in enumerate(edge_indices):
            edge_index = np.concatenate([edge_index, e.numpy() + n_nodes * i], 1)

        score = self.hsic_lasso.run(feats, preds, edge_index.astype(np.int32))
        return score

    @property
    def name(self) -> str:
        return self.hsic_lasso.name + self.perturbator.__class__.__name__

    @property
    def etype(self) -> Literal["node", "edge"]:
        return self._etype
