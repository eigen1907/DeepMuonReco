import torch
import torch.nn as nn
from torch import Tensor
from ..transformers.perceiver import PerceiverEncoder
from ..transformers.perceiver import PerceiverProcessor
from ..transformers.transformer import TransformerDecoder


__all__ = [
    'MuonDetLatentAttnModel',
]


class MuonDetLatentAttnModel(nn.Module):

    # TODO: docstring
    """
    """

    def __init__(
        self,
        track_dim: int,
        segment_dim: int,
        rechit_dim: int,
        output_dim: int,
        model_dim: int,
        num_heads: int,
        muon_det_processor_num_layers,
        track_encoder_num_layers,
        latent_len: int,
        dropout_p: float = 0.1,
        widening_factor: int = 4,
    ) -> None:
        """
        Args:
            latent_len: number of latent vectors in the encoder for muon detector system measurement embeddings
        """
        super().__init__()

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
            in_features=rechit_dim,
            out_features=model_dim,
        )

        # muon detector system measurements encoder
        self.muon_det_encoder = PerceiverEncoder(
            latent_len=latent_len,
            latent_dim=model_dim,
            num_heads=num_heads,
            use_post_attention_residual=True,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            bias=True,
        )

        self.muon_det_processor = PerceiverProcessor(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=muon_det_processor_num_layers,
            widening_factor=widening_factor,
        )

        # NOTE:
        self.track_encoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=track_encoder_num_layers,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            self_attn=False,
        )

        self.classification_head = nn.Linear(
            in_features=model_dim,
            out_features=output_dim,
        )

    def forward(
        self,
        track: Tensor,
        track_data_mask: Tensor,
        segment: Tensor,
        segment_data_mask: Tensor,
        rechit: Tensor,
        rechit_data_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            track: (N, L_trk, D_trk)
            track_pad_mask: (N, L_trk)
            segment: (N, L_seg, D_seg)
            segment_pad_mask: (N, L_seg)
            rechit: (N, L_rec, D_rechit)
            rechit_pad_mask: (N, L_rec)

        Returns:
            logits: (N, L_trk)
        """
        # NOTE: projection
        track_embed = self.track_embedder(track)
        segment_embed = self.segment_embedder(segment)
        rechit_embed = self.rechit_embedder(rechit)

        # NOTE: muon detector system measurement encoding

        # combine muon detector system measurements
        # embed: (N, L_seg + L_rec, D_model)
        muon_det_embed = torch.cat(
            tensors=[
                segment_embed,
                rechit_embed,
            ],
            dim=1,
        )

        # memory pad_mask: (N, L_seg + L_rec)
        muon_det_data_mask = torch.cat(
            tensors=[
                segment_data_mask,
                rechit_data_mask
            ],
            dim=1
        )

        muon_det_embed = self.muon_det_encoder(
            input=muon_det_embed,
            data_mask=muon_det_data_mask,
        )

        muon_det_embed = self.muon_det_processor(
            latent=muon_det_embed,
        )

        # NOTE: tracker track encoding
        track_embed = self.track_encoder(
            target=track_embed,
            source=muon_det_embed,
            target_data_mask=None,
            source_data_mask=None,
        )

        # NOTE: classification head
        #
        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        logits: Tensor = self.classification_head(track_embed)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        logits = logits.squeeze(dim=2)
        # logits = logits.masked_fill(mask=~track_data_mask, value=0)
        return logits
