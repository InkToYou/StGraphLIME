import argparse
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn.functional import softmax
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.node import MixNodeFeatureZeroPerturbator
from hsic_explainer.utils import edge_score2node_score, fix_seed


def explain(
    model_name: str,
    dataloader: DataLoader,
    explainers: Sequence[BaseExplainer],
) -> None:
    fid: dict = {e.name: [] for e in explainers}
    for G in tqdm(dataloader, desc="explain steps", leave=False):
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
            ori_p = (
                softmax(model(x=G.x, edge_index=G.edge_index, batch=G.batch), dim=-1)
                .squeeze()[G.y.item()]
                .item()
            )

            nnz = len(score[score > 0])
            rm_nodes = np.argsort(score)[::-1].copy()[
                : max(1, min(nnz, int(G.num_nodes * 0.4)))
            ]
            pb_vec = torch.ones(G.x.shape)
            pb_vec[rm_nodes] = 0.0
            pb_p = (
                softmax(
                    model(
                        x=G.x * pb_vec,
                        edge_index=G.edge_index,
                        batch=G.batch,
                    ),
                    dim=-1,
                )
                .squeeze()[G.y.item()]
                .item()
            )
            if np.argmax(ori_p) == G.y.item():
                fid[e.name].append(ori_p - pb_p)
    for k in fid:
        print(f"{k} Fidelity: {np.mean(fid[k])}")


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    fix_seed(args.seed)

    path = (Path(__file__) / ".." / "data" / "TU").resolve()
    dataset = TUDataset(path, name="MUTAG")

    dataset = list(dataset)
    print(f"Total data size: {len(dataset)}")
    dataset = dataset[:10]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_pb_data, n_pb = 200, 5
    reg_space = np.logspace(-9, 0, 10)
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=MixNodeFeatureZeroPerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=MixNodeFeatureZeroPerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha1=_alpha1, alpha2=_alpha2
            ),
            perturbator=MixNodeFeatureZeroPerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha1, _alpha2 in product(reg_space, reg_space)
    ]
    explain(
        model_name=args.model,
        dataloader=dataloader,
        explainers=explainers,
    )


if __name__ == "__main__":
    main()
