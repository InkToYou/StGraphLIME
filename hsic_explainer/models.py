from typing import List, Optional

import torch
from dig.xgraph.models import GNNBasic
from torch.nn import Linear, Module, ReLU, Sequential
from torch_geometric.nn import (
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric_temporal.nn.recurrent import TGCN

poolings = {
    "max": global_max_pool,
    "add": global_add_pool,
    "mean": global_mean_pool,
}


class BaseGNNModel(GNNBasic):
    def get_node_embeddings(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseTGNNModel(Module):
    def forward(
        self, xs: List[torch.Tensor], edge_indices: List[torch.LongTensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ExampleGIN(BaseGNNModel):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hidden: int, pool_type: str
    ) -> None:
        super(ExampleGIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(
                Linear(dim_hidden, dim_hidden),
                ReLU(),
                Linear(dim_hidden, dim_hidden),
                ReLU(),
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(dim_hidden, dim_hidden),
                ReLU(),
                Linear(dim_hidden, dim_hidden),
                ReLU(),
            )
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(dim_hidden, dim_hidden),
                ReLU(),
                Linear(dim_hidden, dim_hidden),
                ReLU(),
            )
        )

        self.pool_type = pool_type
        self.pooling = poolings[pool_type]

        self.lin1 = Linear(dim_in, dim_hidden)
        self.lin2 = Linear(dim_hidden, dim_out)

        self.node_embeddings: Optional[torch.Tensor] = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        x = self.lin1(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        self.node_embeddings = x.clone().detach()
        x = self.pooling(x, batch)
        z: torch.Tensor = self.lin2(x)
        return z

    def get_node_embeddings(self) -> torch.Tensor:
        assert self.node_embeddings is not None
        return self.node_embeddings

    @property
    def name(self) -> str:
        return super().name + self.pool_type


class ExampleTGCN(BaseTGNNModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        pool_type: str,
    ):
        super(ExampleTGCN, self).__init__()

        self.lin1 = Linear(dim_in, dim_hidden, bias=True)
        self.conv = TGCN(in_channels=dim_hidden, out_channels=dim_hidden)
        self.pool_type = pool_type
        self.pooling = poolings[pool_type]
        self.lin2 = Linear(dim_hidden, dim_out)

    def forward(
        self, xs: List[torch.Tensor], edge_indices: List[torch.LongTensor]
    ) -> torch.Tensor:
        x = self.lin1(xs[0])
        x = torch.squeeze(x)
        h: torch.Tensor = self.conv(X=x, edge_index=edge_indices[0])

        x = self.lin1(xs[1])
        x = torch.squeeze(x)
        h = self.conv(X=x, edge_index=edge_indices[1], H=h)

        x = self.lin1(xs[2])
        x = torch.squeeze(x)
        h = self.conv(X=x, edge_index=edge_indices[2], H=h)

        h = global_max_pool(h, torch.zeros(h.shape[0], dtype=torch.int64))
        h = self.lin2(h)

        return h

    @property
    def name(self) -> str:
        return super().name + self.pool_type
