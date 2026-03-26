import torch.nn as nn


__all__ = ["MultiLayerPerceptron"]


class MultiLayerPerceptron(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        widening_factor: int = 4,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """ """
        hidden_dim = widening_factor * embed_dim

        super().__init__(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim, bias=bias),
            nn.Dropout(p=dropout),
        )
