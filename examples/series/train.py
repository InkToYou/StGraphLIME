import argparse
import pickle
from pathlib import Path
from typing import List

import torch
from data import NUM_CLASSES, NUM_FEATURES
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from hsic_explainer.models import ExampleTGCN
from hsic_explainer.utils import fix_seed


def train(
    model: Module,
    dataset: List[tuple],
    optimizer: Optimizer,
) -> float:
    model.train()

    ce_loss = CrossEntropyLoss()
    total_loss = 0.0
    for xs, edge_indices, y in tqdm(dataset, desc="train steps", leave=False):
        optimizer.zero_grad()
        output = model(xs=xs, edge_indices=edge_indices)
        loss = ce_loss(output, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / len(dataset)


@torch.no_grad()
def test(model: Module, dataset: List[tuple]) -> float:
    model.eval()

    total_acc = 0
    for xs, edge_indices, y in tqdm(dataset, desc="test steps", leave=False):
        out = model(xs=xs, edge_indices=edge_indices)
        total_acc += int((out.argmax(-1) == y).sum())
    return total_acc / len(dataset)


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool", type=str, required=True, choices=["max", "add", "mean"]
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    fix_seed(args.seed)

    with open(Path(__file__).parent / "trn_data.data", "rb") as f:
        dataset = pickle.load(f)

    trn_rate = 0.9
    n_trn = int(len(dataset) * trn_rate)
    train_dataset = dataset[:n_trn]
    test_dataset = dataset[n_trn:]

    model = ExampleTGCN(NUM_FEATURES, NUM_CLASSES, 128, args.pool)
    optimizer = Adam(model.parameters(), lr=0.001)

    with tqdm(range(1, 201), desc="Epoch") as t:
        for _ in t:
            loss = train(model=model, dataset=train_dataset, optimizer=optimizer)
            train_acc = test(model=model, dataset=train_dataset)
            test_acc = test(model=model, dataset=test_dataset)
            trn_result = {
                "Loss": f"{loss:.4f}",
                "Train Acc": f"{train_acc:.4f}",
                "Test Acc": f"{test_acc:.4f}",
            }
            t.set_postfix(trn_result)
            t.update()

    torch.save(
        model,
        (
            Path(__file__).parent
            / f"{model.name}_{trn_rate}_{args.pool}_{args.seed}.model"
        ).resolve(),
    )
    print("Train results")
    print("-" * 100)
    print(trn_result)
    print("-" * 100)


if __name__ == "__main__":
    main()
