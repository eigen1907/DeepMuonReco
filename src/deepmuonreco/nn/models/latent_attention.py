import torch
import torch.nn as nn
from torch import Tensor
from ..transformers.perceiver import PerceiverEncoder
from ..transformers.transformer import TransformerDecoder


__all__ = [
    'LatentAttentionModel',
]


class LatentAttentionModel(nn.Module):

    # TODO: docstring
    """
    """

    def __init__(
        self,
        track_dim: int,
        segment_dim: int,
        hit_dim: int,
        output_dim: int,

        model_dim: int,
        num_heads: int,
        track_latent_len: int,
        muon_det_latent_len: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        dropout_p: float = 0.1,
        widening_factor: int = 4,
    ) -> None:
        """
        Args:
            latent_len: number of latent vectors in the encoder for muon detector system measurement embeddings
        """
        super().__init__()

        # tracker track
        self.track_embedder = nn.Linear(in_features=track_dim, out_features=model_dim)
        # muon detector
        self.segment_embedder = nn.Linear(in_features=segment_dim, out_features=model_dim)
        self.hit_embedder = nn.Linear(in_features=hit_dim, out_features=model_dim)

        self.track_encoder = PerceiverEncoder(
            latent_len=track_latent_len,
            latent_dim=model_dim,
            num_heads=num_heads,
            use_post_attention_residual=True,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            bias=True,
        )

        self.muon_det_encoder = PerceiverEncoder(
            latent_len=muon_det_latent_len,
            latent_dim=model_dim,
            num_heads=num_heads,
            use_post_attention_residual=True,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            bias=True,
        )

        self.encoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=encoder_num_layers,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            self_attn=False,
        )

        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=decoder_num_layers,
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
        hit_embed = self.hit_embedder(rechit)

        # NOTE: muon detector system measurement encoding

        # combine muon detector system measurements
        # embed: (N, L_seg + L_rec, D_model)
        muon_det_embed = torch.cat(
            tensors=[
                segment_embed,
                hit_embed,
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

        # NOTE: muonn detector measurement encoding
        muon_det_latent = self.muon_det_encoder.forward(
            input=muon_det_embed,
            data_mask=muon_det_data_mask,
        )

        # NOTE: tracker track encoding
        track_latent = self.track_encoder.forward(
            input=track_embed,
            data_mask=track_data_mask,
        )

        latent = self.encoder.forward(
            target=track_latent,
            source=muon_det_latent,
            target_data_mask=None,
            source_data_mask=None,
        )

        track_embed = self.decoder.forward(
            target=track_embed,
            source=latent,
            target_data_mask=track_data_mask,
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
