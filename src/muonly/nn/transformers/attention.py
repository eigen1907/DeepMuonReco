import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops as eo


__all__ = [
    'CrossAttention',
    'SelfAttention',
]


class _Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
    ) -> None:
        """ """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self._dropout = dropout

        self.out_proj = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )

        self.out_dropout = nn.Dropout(
            p=dropout,
        )

    def _forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """ """
        query, key, value = [
            eo.rearrange(each, "n l (h d) -> n h l d", h=self.num_heads)
            for each in [query, key, value]
        ]
        if attn_mask is not None:
            attn_mask = eo.repeat(
                tensor=attn_mask,
                pattern="n t s -> n h t s",
                h=self.num_heads,
            )

        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=(self._dropout if self.training else 0),
        )

        output = eo.rearrange(
            tensor=output,
            pattern="n h t d -> n t (h d)",
        )

        output = self.out_proj(output)
        output = self.out_dropout(output)
        return output


class CrossAttention(_Attention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
    ) -> None:
        """ """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.q_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )
        self.kv_proj = nn.Linear(
            in_features=embed_dim, out_features=(2 * embed_dim), bias=bias
        )

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """ """
        query = self.q_proj(target)
        key, value = self.kv_proj(source).chunk(chunks=2, dim=-1)

        return self._forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )


class SelfAttention(_Attention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
    ) -> None:
        """ """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.qkv_proj = nn.Linear(
            in_features=embed_dim,
            out_features=(3 * embed_dim),
            bias=bias,
        )

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """ """
        query, key, value = self.qkv_proj(input).chunk(chunks=3, dim=-1)
        return self._forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )
