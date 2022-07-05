import argparse
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from hsic_explainer.models import ExampleGIN
from hsic_explainer.utils import fix_seed


def train(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
) -> float:
    model.train()

    ce_loss = CrossEntropyLoss()
    total_loss = 0.0
    for G in tqdm(dataloader, desc="train steps", leave=False):
        optimizer.zero_grad()
        output = model(G.x, G.edge_index, G.batch)
        loss = ce_loss(output, G.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def test(model: Module, dataloader: DataLoader) -> float:
    model.eval()

    total_acc = 0
    for G in tqdm(dataloader, desc="test steps", leave=False):
        out = model(G.x, G.edge_index, G.batch)
        total_acc += int((out.argmax(-1) == G.y).sum())
    return total_acc / len(dataloader.dataset)


def main() -> None:
    fix_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool", type=str, required=True, choices=["max", "add", "mean"]
    )
    args = parser.parse_args()

    path = (Path(__file__).parent / "data").resolve()
    dataset = TUDataset(path, name="MUTAG").shuffle()

    trn_rate = 0.9
    n_trn = int(len(dataset) * trn_rate)
    train_dataset = dataset[:n_trn]
    test_dataset = dataset[n_trn:]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = ExampleGIN(dataset.num_features, dataset.num_classes, 128, args.pool)
    optimizer = Adam(model.parameters(), lr=0.001)

    with tqdm(range(1, 1001), desc="Epoch") as t:
        for _ in t:
            loss = train(model=model, dataloader=train_loader, optimizer=optimizer)
            train_acc = test(model=model, dataloader=train_loader)
            test_acc = test(model=model, dataloader=test_loader)
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
            Path(__file__).parent / f"{model.name}_{trn_rate}_{args.pool}.model"
        ).resolve(),
    )
    print("Train results")
    print("-" * 100)
    print(trn_result)
    print("-" * 100)


if __name__ == "__main__":
    main()
