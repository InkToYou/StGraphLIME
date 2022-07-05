import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.baselines import (
    OcclusionNodeFeatureExplainer,
    RandomNodeExplainer,
)
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.series import RandomWalkNodeFeatureSeriesPerturbator


def explain(
    model_name: str,
    dataset: List[tuple],
    explainers: List[BaseExplainer],
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
            score = e.explain_graph_series(
                xs=xs, edge_indices=edge_indices, model=model, y=y
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    with open(Path(__file__).parent / "trn_data.data", "rb") as f:
        dataset = pickle.load(f)

    n_pb_data, n_pb = 250, 12
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Gauss", pred_kernel="Gauss", alpha=1.0e-09
            ),
            perturbator=RandomWalkNodeFeatureSeriesPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        ),
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Gauss", pred_kernel="Gauss", alpha=1.0e-04
            ),
            perturbator=RandomWalkNodeFeatureSeriesPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        ),
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Gauss", pred_kernel="Gauss", alpha1=1.0, alpha2=1.0e-04
            ),
            perturbator=RandomWalkNodeFeatureSeriesPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        ),
        RandomNodeExplainer(),
        OcclusionNodeFeatureExplainer(),
    ]
    _ = explain(
        model_name=args.model,
        dataset=dataset,
        explainers=explainers,
    )


if __name__ == "__main__":
    main()
