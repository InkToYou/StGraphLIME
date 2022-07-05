import argparse
import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.series import RandomWalkNodeFeatureSeriesPerturbator
from hsic_explainer.utils import fix_seed


def explain(
    model_name: str,
    dataset: List[tuple],
    explainers: Sequence[BaseExplainer],
) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    results: Dict[str, List[Tuple[np.ndarray, int]]] = {e.name: [] for e in explainers}
    p_tot = 0
    p_cor = {k: 0.0 for k in results}
    for xs, edge_indices, y in tqdm(dataset, desc="explain steps", leave=False):
        if y.item() == 0:
            continue
        p_tot += 1
        for e in explainers:
            model = torch.load((Path(__file__).parent / model_name).resolve())
            model.eval()

            try:
                score = e.explain_graph_series(
                    xs=xs, edge_indices=edge_indices, model=model, y=y
                )
            except AssertionError:
                score = (
                    np.zeros(xs[0].shape[0] * len(xs))
                    if e.etype == "node"
                    else np.zeros(edge_indices[0].shape[1] * len(xs))
                )

            tar = np.array([])
            for i, x in enumerate(xs):
                (tmp,) = np.where(x.numpy().squeeze() == 1.0)
                if len(tar) < len(tmp):
                    tar = tmp + i * x.shape[0]
            if score.max() == 0.0:
                can_nodes = [-1]  # Nothing selected
            else:
                can_nodes = score.argsort()[::-1].copy()[: len(tar)]
            p_cor[e.name] += float(len(np.intersect1d(tar, can_nodes)) / len(can_nodes))
            results[e.name].append((score, y.item()))
    for k in results:
        print(f"{k} Acc.: {p_cor[k] / p_tot}")
    return results


def main() -> None:
    fix_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    with open(Path(__file__).parent / "val_data.data", "rb") as f:
        dataset = pickle.load(f)

    n_pb_data, n_pb = 250, 12
    reg_space = np.logspace(-9, 0, 10)
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Gauss", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=RandomWalkNodeFeatureSeriesPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Gauss", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=RandomWalkNodeFeatureSeriesPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Gauss", pred_kernel="Gauss", alpha1=_alpha1, alpha2=_alpha2
            ),
            perturbator=RandomWalkNodeFeatureSeriesPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        )
        for _alpha1, _alpha2 in product(reg_space, reg_space)
    ]
    _ = explain(
        model_name=args.model,
        dataset=dataset,
        explainers=explainers,
    )


if __name__ == "__main__":
    main()
