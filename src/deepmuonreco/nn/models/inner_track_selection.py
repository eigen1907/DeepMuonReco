import torch
import torch.nn as nn
from torch import Tensor
from ..utils import make_cross_attn_mask, make_self_attn_mask
from deepmuonreco.data import InnerTrackSelectionDataset


class InnerTrackSelectionTransformer(nn.Module):

    def __init__(
        self,
        dim_track: int = InnerTrackSelectionDataset.DIM_TRACK,
        dim_seg: int = InnerTrackSelectionDataset.DIM_SEG,
        dim_hit: int = InnerTrackSelectionDataset.DIM_HIT,
        dim_output: int = InnerTrackSelectionDataset.DIM_TARGET,
        dim_model: int = 64,
        dim_feedforward: int = 128,
        activation: str = 'relu',
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        # Track embedding
        self.track_embedder = nn.Linear(
            in_features=dim_track,
            out_features=dim_model,
        )

        # DT/CSC seg embedding
        self.seg_embedder = nn.Linear(
            in_features=dim_seg,
            out_features=dim_model,
        )

        # RPC/GEM hit embedding
        self.hit_embedder = nn.Linear(
            in_features=dim_hit,
            out_features=dim_model,
        )

        layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
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
            in_features=dim_model,
            out_features=dim_output,
        )

    def forward(
        self,
        track: Tensor,
        track_pad_mask: Tensor,
        seg: Tensor,
        seg_pad_mask: Tensor,
        hit: Tensor,
        hit_pad_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            track: (N, L_trk, D_trk)
            track_pad_mask: (N, L_trk)
            seg: (N, L_seg, D_seg)
            seg_pad_mask: (N, L_seg)
            hit: (N, L_rec, D_hit)
            hit_pad_mask: (N, L_rec)

        Returns:
            logits: (N, L_trk)
        """
        # embed track features: (N, L_trk, D_model)
        track_embed = self.track_embedder(track)
        # embed seg features (DT/CSC): (N, L_seg, D_model)
        seg_embed = self.seg_embedder(seg)
        # embed hit features (RPC/GEM): (N, L_rec, D_model)
        hit_embed = self.hit_embedder(hit)
        # concatenate seg and hit dimension -> memory tensor
        # embed: (N, L_seg + L_rec, D_model)
        memory_embed = torch.cat([seg_embed, hit_embed], dim=1)
        # memory pad_mask: (N, L_seg + L_rec)
        memory_pad_mask = torch.cat([seg_pad_mask, hit_pad_mask], dim=1)

        # compute self-attention mask for track features (target)
        target_mask = make_self_attn_mask(
            pad_mask=track_pad_mask,
            num_heads=self.num_heads,
        )

        # compute cross-attention mask between track (target) and combined memory
        memory_mask = make_cross_attn_mask(
            source_pad_mask=memory_pad_mask,
            target_pad_mask=track_pad_mask,
            num_heads=self.num_heads,
        )

        # Transformer decoder: track_embed attends to memory_embed (DT/CSC seg + RPC/GEM hit)
        latent = self.backbone(
            tgt=track_embed,
            memory=memory_embed,
            tgt_mask=target_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_key_padding_mask=track_pad_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )

        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        logits: Tensor = self.classification_head(latent)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        logits = logits.squeeze(dim=2)
        logits = logits.masked_fill(mask=track_pad_mask, value=0)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        return logits