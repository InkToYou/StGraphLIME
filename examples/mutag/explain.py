import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.baselines import GNNExplainer, PGExplainer, SubgraphX
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.node import RandomWalkNodeFeatureZeroPerturbator
from hsic_explainer.utils import edge_score2node_score, fix_seed, tonumpy_copy


def explain(
    model_name: str,
    dataloader: DataLoader,
    explainers: Sequence[BaseExplainer],
) -> Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]]:
    results: Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]] = {
        e.name: [] for e in explainers
    }
    for G in tqdm(dataloader, desc="explain steps", leave=False):
        for e in explainers:
            model = torch.load((Path(__file__).parent / model_name).resolve())
            model.eval()
            score = e.explain_graph(G=G, model=model)
            if e.etype == "edge":
                score = edge_score2node_score(G=G, edge_score=score)
            ori_p = tonumpy_copy(
                softmax(
                    model(x=G.x, edge_index=G.edge_index, batch=G.batch), dim=-1
                ).squeeze()
            )
            results[e.name].append(
                (G, score, ori_p, G.y.item(), np.argmax(ori_p) == G.y.item())
            )
    return results


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-size", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    fix_seed(args.seed)

    path = (Path(__file__) / ".." / "data" / "TU").resolve()
    dataset = TUDataset(path, name="MUTAG")

    dataset = list(dataset)
    dataset = dataset[10:]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_pb_data, n_pb = 200, 5
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=1.0e-08
            ),
            perturbator=RandomWalkNodeFeatureZeroPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        ),
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=1.0e-05
            ),
            perturbator=RandomWalkNodeFeatureZeroPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        ),
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha1=1.0e-04, alpha2=1.0e-01
            ),
            perturbator=RandomWalkNodeFeatureZeroPerturbator(
                n_pb_data=n_pb_data, n_pb=n_pb
            ),
            etype="node",
        ),
        GNNExplainer(),
        PGExplainer(),
        SubgraphX(max_size=args.max_size, subgraph_building_method="zero_filling"),
    ]
    results = explain(
        model_name=args.model,
        dataloader=dataloader,
        explainers=explainers,
    )

    with open(
        (
            Path(__file__).parent
            / f"results_{args.model}_{args.max_size}_{n_pb}_{n_pb_data}_{args.seed}.pkl"
        ).resolve(),
        "wb",
    ) as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
