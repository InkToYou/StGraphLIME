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
from hsic_explainer.perturbators.edge import EdgeRemovePerturbator
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
    p_cor2 = {k: 0 for k in results}
    p_cor5 = {k: 0 for k in results}
    for G in tqdm(dataloader, desc="explain steps", leave=False):
        if G.y.item() == 0:
            continue
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
                can_nodes5 = [-1]  # Nothing selected
            else:
                can_nodes2 = score.argsort()[::-1].copy()[:2]
                can_nodes5 = score.argsort()[::-1].copy()[:5]
            tar = degree(G.edge_index[0]).argsort(descending=True)[:2].numpy()
            p_cor2[e.name] += int(tar[0] in can_nodes2 and tar[1] in can_nodes2)
            p_cor5[e.name] += int(tar[0] in can_nodes5 and tar[1] in can_nodes5)
            results[e.name].append(
                (G, score, ori_p, G.y.item(), np.argmax(ori_p) == G.y.item())
            )
    for k in results:
        print(f"{k} 2Acc.: {p_cor2[k] / p_tot}")
        print(f"{k} 5Acc.: {p_cor5[k] / p_tot}")
    return results


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    dataset = gen_exp_data()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_pb_data, n_pb = 200, 3
    explainers = [
        HSICExplainer(
            hsic_lasso=HSICLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=1.0e-07
            ),
            perturbator=EdgeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="edge",
        ),
        HSICExplainer(
            hsic_lasso=HSICLatentGroupLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha=1.0e-02
            ),
            perturbator=EdgeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="edge",
        ),
        HSICExplainer(
            hsic_lasso=HSICFusedLasso(
                feat_kernel="Delta", pred_kernel="Gauss", alpha1=1.0, alpha2=1.0
            ),
            perturbator=EdgeRemovePerturbator(n_pb_data=n_pb_data, n_pb=n_pb),
            etype="edge",
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
