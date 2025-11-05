import warnings
import torch
import torch.nn as nn
from torch import Tensor
from ..utils import make_cross_attn_mask, make_self_attn_mask


__all__ = [
    'LegacyVanillaTransformerModel',
]


class LegacyVanillaTransformerModel(nn.Module):

    def __init__(
        self,
        track_dim: int,
        segment_dim: int,
        hit_dim: int,
        output_dim: int,
        model_dim: int = 64,
        feedforward_dim: int = 128,
        activation: str = 'relu',
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """
        """
        warnings.warn(
            message=(
                "LegacyVanillaTransformerModel is deprecated and will be removed in future releases. "
                "Please use the updated VanillaTransformerModel instead."
            ),
            category=DeprecationWarning,
        )

        super().__init__()

        self.num_heads = num_heads

        # Track embedding (px, py, eta)
        self.track_embedder = nn.Linear(
            in_features=track_dim,
            out_features=model_dim,
        )

        # DT/CSC segment embedding (position + direction; dimension: 6)
        self.segment_embedder = nn.Linear(
            in_features=segment_dim,
            out_features=model_dim,
        )

        # RPC/GEM rechit embedding (position only; dimension: 3)
        self.rechit_embedder = nn.Linear(
            in_features=hit_dim,
            out_features=model_dim,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
            bias=True,
        )

        self.backbone = nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers,
        )

        self.classification_head = nn.Linear(
            in_features=model_dim,
            out_features=output_dim,
        )

    def forward(
        self,
        tracker_track: Tensor,
        tracker_track_data_mask: Tensor,
        dt_segment: Tensor,
        dt_segment_data_mask: Tensor,
        csc_segment: Tensor,
        csc_segment_data_mask: Tensor,
        rpc_hit: Tensor,
        rpc_hit_data_mask: Tensor,
        gem_hit: Tensor,
        gem_hit_data_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            tracker_track: (N, L_trk, D_trk)
            tracker_track_data_mask: (N, L_trk)
            dt_segment: (N, L_dt_seg, D_dt_seg)
            dt_segment_data_mask: (N, L_dt_seg)
            csc_segment: (N, L_csc_seg, D_csc_seg)
            csc_segment_data_mask: (N, L_csc_seg)
            rpc_hit: (N, L_rpc_rec, D_rpc_rec)
            rpc_hit_data_mask: (N, L_rpc_rec)
            gem_hit: (N, L_gem_rec, D_gem_rec)
            gem_hit_data_mask: (N, L_gem_rec)
        Returns:
            logits: (N, L_trk)
        """
        if (dt_segment.size(2) != csc_segment.size(2)) or (rpc_hit.size(2) != gem_hit.size(2)):
            raise ValueError("Segment and rechit feature dimensions must match respectively.")

        segment = torch.cat([dt_segment, csc_segment], dim=1)
        segment_data_mask = torch.cat([dt_segment_data_mask, csc_segment_data_mask], dim=1)
        rechit = torch.cat([rpc_hit, gem_hit], dim=1)
        rechit_data_mask = torch.cat([rpc_hit_data_mask, gem_hit_data_mask], dim=1)

        # invert data mask to get pad mask
        tracker_track_pad_mask = ~tracker_track_data_mask
        segment_pad_mask = ~segment_data_mask
        rechit_pad_mask = ~rechit_data_mask

        # embed track features: (N, L_trk, D_model)
        track_embed = self.track_embedder(tracker_track)
        # embed segment features (DT/CSC): (N, L_seg, D_model)
        segment_embed = self.segment_embedder(segment)
        # embed rechit features (RPC/GEM): (N, L_rec, D_model)
        rechit_embed = self.rechit_embedder(rechit)
        # concatenate segment and rechit dimension -> memory tensor
        # embed: (N, L_seg + L_rec, D_model)
        memory_embed = torch.cat([segment_embed, rechit_embed], dim=1)
        # memory pad_mask: (N, L_seg + L_rec)
        memory_pad_mask = torch.cat([segment_pad_mask, rechit_pad_mask], dim=1)

        # compute self-attention mask for track features (target)
        tgt_mask = make_self_attn_mask(
            pad_mask=tracker_track_pad_mask,
            num_heads=self.num_heads,
        )

        # compute cross-attention mask between track (target) and combined memory
        memory_mask = make_cross_attn_mask(
            source_pad_mask=memory_pad_mask,
            target_pad_mask=tracker_track_pad_mask,
            num_heads=self.num_heads,
        )

        # Transformer decoder: track_embed attends to memory_embed (DT/CSC segment + RPC/GEM rechit)
        latent = self.backbone(
            tgt=track_embed,
            memory=memory_embed,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_key_padding_mask=tracker_track_pad_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )

        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        logits: Tensor = self.classification_head(latent)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        logits = logits.squeeze(dim=2)
        logits = logits.masked_fill(mask=tracker_track_pad_mask, value=0)
        return logits
