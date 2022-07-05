import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from hsic_explainer.utils import draw_graph, fix_seed

atoms = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}


def visualize(results: dict, max_size: float) -> None:
    n_res = len(results[list(results.keys())[0]])
    Gs = [results[list(results.keys())[0]][i][0] for i in range(n_res)]
    preds = [results[list(results.keys())[0]][i][2] for i in range(n_res)]
    ys = [results[list(results.keys())[0]][i][3] for i in range(n_res)]
    correct = [results[list(results.keys())[0]][i][4] for i in range(n_res)]
    idx = sorted(
        np.arange(n_res), key=lambda i: -preds[i][ys[i]] * int(correct[i]) * (1 - ys[i])
    )
    n_nodes: List[Optional[int]] = [None] * len(results[list(results.keys())[0]])
    for i, (_, score, _, _, _) in enumerate(results["SubgraphX"]):
        (rm_nodes,) = np.where(score == 1.0)
        n_nodes[i] = len(rm_nodes)

    for i, _ in zip(idx, range(4)):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for j, k in enumerate(results.keys()):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            score = results[k][i][1].copy()

            nnz = len(score[score > 0])
            if "SubgraphX" in k:
                score = score
                n_nodes[i] = len(score[score == 1.0])
            else:
                n_imp = n_nodes[i]
                n_imp = int(Gs[i].num_nodes * max_size)
                assert n_imp is not None and n_imp > 0
                rm_nodes = np.argsort(score)[::-1].copy()[: max(1, min(nnz, n_imp))]
                score = np.zeros(Gs[i].num_nodes)
                score[rm_nodes] = 1.0

            x = Gs[i].x.argmax(dim=1).numpy()
            labels = {a: atoms[b] for a, b in enumerate(x)}
            draw_graph(
                g=to_networkx(Gs[i], to_undirected=True),
                ax=ax,
                title="",
                labels=labels,
                with_labels=True,
                node_color=score,
            )
            print(f"Pred: {preds[i][ys[i]]:.3f}, Label: {ys[i]}, N: {i}")

            vis_dir = Path(__file__).parent / "figure"
            vis_dir.mkdir(exist_ok=True)
            plt.savefig(vis_dir / f"max_explanations_{i}_{k}_labal_{ys[i]}.png")
            plt.clf()


def main() -> None:
    fix_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        dest="results",
        type=Path,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--max-size",
        dest="max_size",
        type=float,
        required=True,
    )
    args = parser.parse_args()

    results = {}
    for res in args.results:
        with open(res, "rb") as f:
            data = pickle.load(f)
            results.update(data)
    for k in list(results.keys()):
        if "HSICLasso" in k:
            results["HSICLasso"] = results.pop(k)
        elif "HSICLatentGroupLasso" in k:
            results["HSICGroupLasso"] = results.pop(k)
        elif "HSICFusedLasso" in k:
            results["HSICFusedLasso"] = results.pop(k)
        elif "SubgraphX" in k:
            results["SubgraphX"] = results.pop(k)

    visualize(results=results, max_size=args.max_size)


if __name__ == "__main__":
    main()
