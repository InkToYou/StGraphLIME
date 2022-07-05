import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.nn.functional import softmax

from hsic_explainer.utils import fix_seed


@torch.no_grad()
def eval(model: Module, results: dict, max_size: float) -> dict:
    model.eval()
    scores: dict = {k: [] for k in results}
    n_nodes: List[Optional[int]] = [None] * len(results[list(results.keys())[0]])
    for i, (_, score, _, _, _) in enumerate(results["SubgraphX"]):
        (rm_nodes,) = np.where(score == 1.0)
        n_nodes[i] = len(rm_nodes)

    for k, res in results.items():
        for i, (G, score, p, y, correct) in enumerate(res):
            if not correct:
                continue
            ori_p = (
                softmax(model(x=G.x, edge_index=G.edge_index, batch=G.batch), dim=-1)
                .squeeze()[y]
                .item()
            )
            assert p[y] == ori_p, f"{p[y]}, {ori_p}"
            assert (np.argmax(p) == y) == correct, f"{np.argmax(p)}, {y}, {correct}"

            nnz = len(score[score > 0.0])
            if k == "SubgraphX":
                (rm_nodes,) = np.where(score == 1.0)
            else:
                n_imp = n_nodes[i]
                # assert n_imp is not None and n_imp > 0
                n_imp = int(G.num_nodes * max_size)
                rm_nodes = np.argsort(score)[::-1].copy()[: max(1, min(nnz, n_imp))]

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
                .squeeze()[y]
                .item()
            )
            scores[k].append(
                (
                    ori_p - pb_p,
                    1 - (len(rm_nodes) / G.num_nodes),
                )
            )
    return scores


def main() -> None:
    fix_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        dest="model",
        type=Path,
        required=True,
    )
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

    model = torch.load(args.model)
    results = {}
    for res in args.results:
        with open(res, "rb") as f:
            data = pickle.load(f)
            results.update(data)
            if "SubgraphX" in data:
                results["SubgraphX"] = data["SubgraphX"]
            else:
                results.update(data)

    for k in list(results.keys()):
        print(k)
        if "HSICLasso" in k:
            results["HSICLasso"] = results.pop(k)
        elif "HSICLatentGroupLasso" in k:
            results["HSICGroupLasso"] = results.pop(k)
        elif "HSICFusedLasso" in k:
            results["HSICFusedLasso"] = results.pop(k)
        elif "SubgraphX" in k:
            results["SubgraphX"] = results.pop(k)

    scores = eval(model=model, results=results, max_size=args.max_size)

    fidelity: dict = {k: [] for k in scores}
    sparsity: dict = {k: [] for k in scores}

    for k, v in scores.items():
        for f, s in v:
            fidelity[k].append(f)
            sparsity[k].append(s)

    for k in scores:
        idx = np.argsort(sparsity[k])
        fidelity[k] = np.array(fidelity[k])[idx]
        sparsity[k] = np.array(sparsity[k])[idx]

    for k in scores:
        _fd = []
        _sp = []
        i = 0
        t = sparsity[k][0]
        tmp = [fidelity[k][0]]
        while i < len(fidelity[k]):
            if t == sparsity[k][i]:
                tmp.append(fidelity[k][i])
            else:
                _fd.append(np.mean(tmp))
                _sp.append(t)
                t = sparsity[k][i]
                tmp = [fidelity[k][i]]
            i += 1
        sparsity[k] = _sp
        fidelity[k] = _fd

    num = 10
    b = np.ones(num) / num
    trunc = int(num / 2)
    params = {
        "SubgraphX": ("deepskyblue", "dashed"),
        "GNNExplainer": ("deepskyblue", "dotted"),
        "PGExplainer": ("deepskyblue", "dashdot"),
        "HSICLasso": ("orange", "dashed"),
        "HSICGroupLasso": ("orange", "dotted"),
        "HSICFusedLasso": ("orange", "dashdot"),
    }

    for k in scores:
        plt.plot(
            sparsity[k][trunc:-trunc],
            np.convolve(fidelity[k], b, mode="same")[trunc:-trunc],
            c=params[k][0],
            ls=params[k][1],
            label=k,
        )
    plt.hlines(0, 0, 1, alpha=0.5, colors="black")
    plt.ylim(-0.1, 1)
    plt.xlim(0.7, 0.85)
    plt.xlabel("Sparsity")
    plt.ylabel("Fidelity")
    plt.title("Fidelity GIN + mean pooling")
    plt.legend()
    plt.savefig(Path(__file__).parent / "fidelity.png")
    plt.clf()

    for k in scores:
        print(k, np.mean(fidelity[k]), np.mean(sparsity[k]))


if __name__ == "__main__":
    main()
