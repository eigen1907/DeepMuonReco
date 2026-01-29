import torch
import torch.nn as nn
from torch import Tensor
from ..utils import make_cross_attn_mask, make_self_attn_mask


__all__ = [
    'VanillaTransformerModel',
]


class VanillaTransformerModel(nn.Module):

    def __init__(
        self,
        # input dimensions
        tracker_track_dim: int,
        dt_segment_dim: int,
        csc_segment_dim: int,
        gem_segment_dim: int,
        rpc_hit_dim: int,
        gem_hit_dim: int,
        # output dimensions
        output_dim: int,
        # model hyperparameters
        model_dim: int = 64,
        feedforward_dim: int = 128,
        activation: str = 'relu',
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        # tracker track: tt
        self.tracker_track_embedder = nn.Linear(in_features=tracker_track_dim, out_features=model_dim)

        # muon detector measurements: mdm
        self.dt_segment_embedder = nn.Linear(in_features=dt_segment_dim, out_features=model_dim)
        self.csc_segment_embedder = nn.Linear(in_features=csc_segment_dim, out_features=model_dim)
        self.gem_segment_embedder = nn.Linear(in_features=gem_segment_dim, out_features=model_dim)
        self.rpc_hit_embedder = nn.Linear(in_features=rpc_hit_dim, out_features=model_dim)
        self.gem_hit_embedder = nn.Linear(in_features=gem_hit_dim, out_features=model_dim)

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
        gem_segment: Tensor,
        gem_segment_data_mask: Tensor,
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
            gem_segment: (N, L_gem_rec, D_gem_seg)
            gem_segment_data_mask: (N, L_gem_seg)
            rpc_hit: (N, L_rpc_rec, D_rpc_rec)
            rpc_hit_data_mask: (N, L_rpc_rec)
            gem_hit: (N, L_gem_rec, D_gem_rec)
            gem_hit_data_mask: (N, L_gem_rec)
        Returns:
            logits: (N, L_trk)
        """
        # NOTE: projection
        tracker_track_embed = self.tracker_track_embedder(tracker_track)
        dt_segment_embed = self.dt_segment_embedder(dt_segment)
        csc_segment_embed = self.csc_segment_embedder(csc_segment)
        gem_segment_embed = self.gem_segment_embedder(gem_segment)
        rpc_hit_embed = self.rpc_hit_embedder(rpc_hit)
        gem_hit_embed = self.gem_hit_embedder(gem_hit)

        # NOTE: muon detector system measurement encoding

        # Combine muon detector measurements
        # embed: (N, L_muon_det, D_model)
        # where L_muon_det = L_dt_seg + L_csc_seg + L_gem_seg + L_rpc_hit + L_gem_hit
        muon_det_embed = torch.cat(
            tensors=[
                dt_segment_embed,
                csc_segment_embed,
                gem_segment_embed,
                rpc_hit_embed,
                gem_hit_embed,
            ],
            dim=1, # along sequence length dimension
        )

        muon_det_data_mask = torch.cat(
            tensors=[
                dt_segment_data_mask,
                csc_segment_data_mask,
                gem_segment_data_mask,
                rpc_hit_data_mask,
                gem_hit_data_mask,
            ],
            dim=1 # along sequence length dimension
        )

        tracker_track_pad_mask = ~tracker_track_data_mask
        muon_det_pad_mask = ~muon_det_data_mask

        # compute self-attention mask for track features (target)
        tgt_mask = make_self_attn_mask(
            pad_mask=tracker_track_pad_mask,
            num_heads=self.num_heads,
        )

        # compute cross-attention mask between track (target) and combined memory
        muon_det_mask = make_cross_attn_mask(
            source_pad_mask=muon_det_pad_mask,
            target_pad_mask=tracker_track_pad_mask,
            num_heads=self.num_heads,
        )

        # Transformer decoder: track_embed attends to memory_embed (DT/CSC segment + RPC/GEM rechit)
        latent = self.backbone(
            tgt=tracker_track_embed,
            memory=muon_det_embed,
            tgt_mask=tgt_mask,
            memory_mask=muon_det_mask,
            memory_key_padding_mask=muon_det_pad_mask,
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
