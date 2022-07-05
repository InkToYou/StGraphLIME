import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from data import gen_exp_data
from torch.nn.functional import softmax
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
from tqdm import tqdm

from hsic_explainer.explainers.base import BaseExplainer
from hsic_explainer.explainers.baselines import (
    GNNExplainer,
    PGExplainer,
    RandomNodeExplainer,
)
from hsic_explainer.explainers.hsic import (
    HSICExplainer,
    HSICFusedLasso,
    HSICLasso,
    HSICLatentGroupLasso,
)
from hsic_explainer.perturbators.node import NodeRemovePerturbator
from hsic_explainer.utils import edge_score2node_score, tonumpy_copy


def explain(
    model_name: str,
    dataloader: DataLoader,
    explainers: Sequence[BaseExplainer],
) -> Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]]:
    results: Dict[str, List[Tuple[Data, np.ndarray, np.ndarray, int, bool]]] = {
        e.name: [] for e in explainers
    }
    p_tot = 0
    skip = 0
    and_cor2 = {k: 0 for k in results}
    or_cor2 = {k: 0 for k in results}
    and_cor6 = {k: 0 for k in results}
    or_cor6 = {k: 0 for k in results}
    for G in tqdm(dataloader, desc="explain steps", leave=False):
        if G.y.item() != 1:
            continue
        dg = np.sort(degree(G.edge_index[0]).numpy())[::-1]
        if dg[0] != dg[1]:
            skip += 1
            # print(dg)
            continue
        print(dg, G.num_nodes // 2)
        assert dg[0] == G.num_nodes // 2 - 1
        p_tot += 1
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

            if score.max() == 0.0:
                can_nodes2 = [-1]  # Nothing selected
                can_nodes6 = [-1]  # Nothing selected
            else:
                can_nodes2 = score.argsort()[::-1].copy()[:2]
                can_nodes6 = score.argsort()[::-1].copy()[:6]
            tar = degree(G.edge_index[0]).numpy().argsort()[::-1][:2]
            and_cor2[e.name] += int(tar[0] in can_nodes2 and tar[1] in can_nodes2)
            or_cor2[e.name] += int(tar[0] in can_nodes2 or tar[1] in can_nodes2)
            and_cor6[e.name] += int(tar[0] in can_nodes6 and tar[1] in can_nodes6)
            or_cor6[e.name] += int(tar[0] in can_nodes6 or tar[1] in can_nodes6)
            results[e.name].append(
                (G, score, ori_p, G.y.item(), np.argmax(ori_p) == G.y.item())
            )

    with open(f"./double_results_{model_name}.txt", "a") as f:
        for k in results:
            f.write(f"{k} 2ANDAcc.: {and_cor2[k] / p_tot}\n")
            f.write(f"{k} 2ORAcc.: {or_cor2[k] / p_tot}\n")
            f.write(f"{k} 6ANDAcc.: {and_cor6[k] / p_tot}\n")
            f.write(f"{k} 6ORAcc.: {or_cor6[k] / p_tot}\n")
    print(f"target: {p_tot}, mix: {skip}")
    return results


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    dataset = gen_exp_data()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_pb_data, n_pb = 200, 4
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=1.0e-8
            ),
            perturbator=NodeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        ),
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=1.0e-7
            ),
            perturbator=NodeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        ),
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha1=1.0, alpha2=1.0e-07
            ),
            perturbator=NodeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="node",
        ),
        GNNExplainer(),
        PGExplainer(),
        RandomNodeExplainer(),
    ]
    _ = explain(
        model_name=args.model,
        dataloader=dataloader,
        explainers=explainers,
    )


if __name__ == "__main__":
    main()
