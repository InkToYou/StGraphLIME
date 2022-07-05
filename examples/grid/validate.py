import argparse
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from data import gen_val_data
from torch.nn.functional import softmax
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.node import NodeFeatureNoisePerturbator
from hsic_explainer.utils import edge_score2node_score, fix_seed, tonumpy_copy


def edge_imp2node_imp(G: Data, ed_imp: np.ndarray) -> np.ndarray:
    nd_imp = np.zeros(G.num_nodes)
    for i, (s, t) in zip(ed_imp, G.edge_index.T.numpy()):
        nd_imp[s] = max(nd_imp[s], i)
        nd_imp[t] = max(nd_imp[t], i)

    return nd_imp


def explain(
    model_name: str,
    dataloader: DataLoader,
    explainers: Sequence[BaseExplainer],
) -> Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]]:
    results: Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]] = {
        e.name: [] for e in explainers
    }
    p_tot = 0
    p_cor = {k: 0.0 for k in results}
    for G in tqdm(dataloader, desc="explain steps", leave=False):
        if G.y.item() == 2 or G.y.item() == 3:
            continue
        p_tot += 1
        for e in explainers:
            model = torch.load((Path(__file__).parent / model_name).resolve())
            model.eval()
            try:
                score = e.explain_graph(G=G, model=model)
            except AssertionError:
                score = (
                    np.zeros(G.num_nodes)
                    if e.etype == "node"
                    else np.zeros(G.edge_index.shape[1])
                )
            if e.etype == "edge":
                score = edge_score2node_score(G=G, edge_score=score)
            ori_p = tonumpy_copy(
                softmax(
                    model(x=G.x, edge_index=G.edge_index, batch=G.batch), dim=-1
                ).squeeze()
            )

            if score.max() == 0.0:
                can_nodes = [-1]  # Nothing selected
            else:
                can_nodes = score.argsort()[::-1].copy()[:4]

            f = tonumpy_copy(G.x).flatten()
            (tar,) = np.where(f == 1.0)
            p_cor[e.name] += float(len(np.intersect1d(tar, can_nodes)) / len(can_nodes))
            results[e.name].append(
                (G, score, ori_p, G.y.item(), np.argmax(ori_p) == G.y.item())
            )
    for k in results:
        print(f"{k} Acc.: {p_cor[k] / p_tot}")
    return results


def main() -> None:
    fix_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    dataset = gen_val_data()
    print(f"Total data size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_pb_data, n_pb = 200, 5
    reg_space = np.logspace(-9, 0, 10)
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=NodeFeatureNoisePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=NodeFeatureNoisePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha1=_alpha1, alpha2=_alpha2
            ),
            perturbator=NodeFeatureNoisePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha1, _alpha2 in product(reg_space, reg_space)
    ]
    _ = explain(
        model_name=args.model,
        dataloader=dataloader,
        explainers=explainers,
    )


if __name__ == "__main__":
    main()
