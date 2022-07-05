import argparse
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from data import gen_val_data
from torch.nn.functional import softmax
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.node import NodeRemovePerturbator
from hsic_explainer.utils import edge_score2node_score, fix_seed, tonumpy_copy


def explain(
    model_name: str,
    dataloader: DataLoader,
    explainers: Sequence[BaseExplainer],
    max_size: float,
) -> Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]]:
    results: Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]] = {
        e.name: [] for e in explainers
    }
    p_tot = 0
    p_cor = {k: 0 for k in results}
    b_cor = {k: 0 for k in results}
    for G in tqdm(dataloader, desc="explain steps", leave=False):
        if G.y.item() != 1:
            continue
        dg = np.sort(degree(G.edge_index[0]).numpy())[::-1]
        if dg[0] != dg[1]:
            # print(dg)
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
                selected = (
                    int(max_size * G.num_nodes) if max_size < 1 else int(max_size)
                )
                can_nodes = score.argsort()[::-1].copy()[:selected]
            tar = degree(G.edge_index[0]).numpy().argsort()[::-1][:2]
            p_cor[e.name] += int(tar[0] in can_nodes and tar[1] in can_nodes)
            b_cor[e.name] += int(tar[0] in can_nodes or tar[1] in can_nodes)
            results[e.name].append(
                (G, score, ori_p, G.y.item(), np.argmax(ori_p) == G.y.item())
            )

    for k in results:
        print(f"{k} Acc.: {p_cor[k] / p_tot}")
        print(f"{k} bAcc.: {b_cor[k] / p_tot}")
    return results


def main() -> None:
    fix_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-size", type=float, required=True)
    args = parser.parse_args()

    dataset = gen_val_data()
    print(f"Total data size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_pb_data, n_pb = 200, 4
    reg_space = np.logspace(-9, 0, 10)
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=NodeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=_alpha
            ),
            perturbator=NodeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha in reg_space
    ]

    explainers = explainers + [
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha1=_alpha1, alpha2=_alpha2
            ),
            perturbator=NodeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        )
        for _alpha1, _alpha2 in product(reg_space, reg_space)
    ]

    _ = explain(
        model_name=args.model,
        dataloader=dataloader,
        explainers=explainers,
        max_size=args.max_size,
    )


if __name__ == "__main__":
    main()
