"""Descriptor ANN and graph neural network (GINE) models."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool
from torch_geometric.utils import scatter


class DescriptorANN(nn.Module):
    """Fully connected network: Linear -> LayerNorm -> activation -> dropout per hidden layer."""

    def __init__(self, n_inputs: int, hidden_layers, activation: str, dropout: float):
        super().__init__()
        layers, width = [], n_inputs
        for size in hidden_layers:
            layers += [nn.Linear(width, size), nn.LayerNorm(size),
                       nn.ReLU() if activation == "relu" else nn.GELU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class Ensemble(nn.Module):
    """Mean output of independently initialized models."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.models]).mean(dim=0).unsqueeze(-1)


class GINE(nn.Module):
    """Four GINE layers; mean, sum and max pooling over the input and every layer."""

    def __init__(self, node_dim: int, hidden: int = 128, depth: int = 4, dropout: float = 0.15,
                 edge_dim: int = 5):
        super().__init__()
        self.dropout, self.edge_dim = dropout, edge_dim
        self.input = nn.Linear(node_dim, hidden)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.projection = nn.Sequential(
            nn.Linear(3 * hidden * (depth + 1), 2 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def _edges(self, graph):
        return F.one_hot(graph.edge_type.clamp(0, self.edge_dim - 1), self.edge_dim).to(graph.x.dtype)

    def atom_states(self, graph):
        """Atom representations after the input layer and after each GINE layer."""
        edges = self._edges(graph)
        h = F.gelu(self.input(graph.x))
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, graph.edge_index, edges)
            h = norm(h + F.dropout(F.gelu(update), p=self.dropout, training=self.training))
            states.append(h)
        return states

    def readout(self, pooled):
        return self.head(self.projection(torch.cat(pooled, dim=1))).squeeze(-1)

    def forward(self, graph):
        count = scatter(graph.x.new_ones(graph.num_nodes), graph.batch, dim=0, reduce="sum").clamp_min(1.0)
        pooled = []
        for state in self.atom_states(graph):
            total = global_add_pool(state, graph.batch)
            pooled += [total / count[:, None], total, global_max_pool(state, graph.batch)]
        return self.readout(pooled)

    def masked_readout(self, states, masks):
        """Predictions for one molecule with selected atoms left out of the pooling.

        states: atom states of the unmasked molecule; masks: (variants, atoms), 0 = masked.
        Message passing is unchanged; masked atoms are removed from the sum and the
        maximum, and the mean keeps the full atom count."""
        pooled = []
        for state in states:
            total = masks @ state
            maximum = state[None].expand(len(masks), -1, -1).masked_fill(
                masks[:, :, None] < 0.5, -torch.inf).amax(dim=1)
            pooled += [total / state.shape[0], total, maximum]
        return self.readout(pooled)
