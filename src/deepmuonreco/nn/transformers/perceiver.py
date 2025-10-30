import math
import warnings
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
import einops as eo
from .transformer import CrossAttentionBlock, TransformerEncoderLayer
from .transformer import MLPBlock


__all__ = [
    'PerceiverEncoder',
    'PerceiverProcessor',
    'PerceiverProcessorBlock',
    'PerceiverBasicDecoder',
    'PerceiverLatentQueryDecoder',
]


class PerceiverEncoder(nn.Module):
    """
    cross attention between an input source array and a latent target array
    """

    def __init__(
        self,
        latent_len: int,
        latent_dim: int,
        num_heads: int,
        use_post_attention_residual: bool = True,
        widening_factor: int = 4,
        input_dim: int | None = None,
        dropout_p: float = 0,
        bias: bool = False,
    ) -> None:
        """
        """
        super().__init__()

        self.latent = self._make_latent(latent_len, latent_dim)

        self.attention = CrossAttentionBlock(
            embed_dim=latent_dim,
            num_heads=num_heads,
            use_post_attention_residual=use_post_attention_residual,
            target_dim=latent_dim,
            source_dim=input_dim,
            output_dim=None,
            dropout_p=dropout_p,
            bias=bias,
        )

        self.mlp = MLPBlock(
            embed_dim=latent_dim,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

    def _make_latent(self, latent_len: int, latent_dim: int) -> nn.Parameter:
        """
        adapted from:
            - https://github.com/google-deepmind/hierarchical_perceiver/blob/b3074a4/perceiver_helpers.py#L145-L167
            - https://github.com/google-deepmind/dm-haiku/blob/v0.0.14/haiku/_src/initializers.py#L152-L234
            - https://github.com/google-deepmind/dm-haiku/blob/v0.0.14/haiku/_src/initializers.py#L97C1-L131C28
        """
        latent = torch.empty(latent_len, latent_dim)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(latent)
        scale = 1
        n = max(1, fan_in)
        s = scale / n
        stddev = math.sqrt(s)
        stddev = stddev / .87962566103423978
        nn.init.trunc_normal_(tensor=latent, mean=0, std=stddev, a=-2, b=+2)
        return nn.Parameter(data=latent)


    def forward(
        self,
        input: Tensor,
        data_mask: Tensor | None,
        pre_attention_residual: Tensor | None = None,
    ) -> Tensor:
        """
        """
        batch_size = input.size(0)
        latent_len = self.latent.size(0)

        latent = eo.repeat(
            tensor=self.latent,
            pattern='l d -> n l d',
            n=batch_size,
        )

        if data_mask is None:
            attn_mask = None
        else:
            # n: batch size, s: source array length, t: target array length
            attn_mask = eo.repeat(
                tensor=data_mask,
                pattern='n s -> n t s',
                t=latent_len,
            )

        if pre_attention_residual is not None:
            latent = latent + pre_attention_residual

        output = self.attention(
            target=latent,
            source=input,
            attn_mask=attn_mask,
        )
        output = self.mlp(
            input=output,
        )
        return output


class PerceiverProcessor(TransformerEncoderLayer):


    def forward( # type: ignore[override]
        self,
        latent: Tensor,
    ) -> Tensor:
        """
        """
        return super().forward(input=latent, attn_mask=None)


class PerceiverProcessorBlock(nn.Module):

    def __init__(
        self,
        num_layers: int,
        latent_dim: int,
        num_heads: int,
        widening_factor: int = 4,
        dropout_p: float = 0,
        weight_sharing: bool = False,
    ) -> None:
        """
        """
        super().__init__()
        if num_layers < 0:
            raise ValueError(f'num_layers should be a positive integer. got {num_layers}.')
        elif num_layers == 0:
            warnings.warn('num_layers is 0. the module will be a pass-through.')

        layer = PerceiverProcessor(
            model_dim=latent_dim,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

        if weight_sharing:
            self.layer_list = nn.ModuleList([layer] * num_layers)
        else:
            self.layer_list = _get_clones(module=layer, N=num_layers)

    def forward(
        self,
        latent: Tensor,
    ) -> Tensor:
        """
        """
        output = latent
        for layer in self.layer_list:
            output = layer(latent=output)
        return output


class PerceiverBasicDecoder(nn.Module):
    """Cross-attention-based decoder."""

    def __init__(
        self,
        latent_dim: int,
        query_dim: int | None = None,
        num_heads: int = 1,
        embed_dim: int | None = None,
        widening_factor: int = 1,
        use_post_attention_residual: bool = False,
        dropout_p: float = 0,
    ) -> None:
        """
        Args:
            query_dim: target array
            latent_dim: source array

        Returns:
            N/A
        """
        super().__init__()

        embed_dim = embed_dim or latent_dim
        query_dim = query_dim or latent_dim

        self.attention = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_post_attention_residual=use_post_attention_residual,
            target_dim=query_dim,
            source_dim=latent_dim,
            dropout_p=dropout_p,
        )
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        latent: Tensor,
        query: Tensor,
        query_data_mask: Tensor | None,
    ) -> Tensor:
        """
        """
        output: Tensor = self.attention(
            target=query,
            source=latent,
            attn_mask=None, # FIXME:
        )
        output = self.mlp(
            input=output,
        )

        if query_data_mask is not None:
            pad_mask = query_data_mask.unsqueeze(dim=-1).logical_not()
            output.masked_fill_(mask=pad_mask, value=0)

        return output


class PerceiverLatentQueryDecoder(PerceiverEncoder):
    ...
