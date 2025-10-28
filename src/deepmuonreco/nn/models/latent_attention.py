import torch
import torch.nn as nn
from torch import Tensor
from ..transformers.perceiver import PerceiverEncoder, PerceiverProcessorBlock
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
        # input dimensions
        tracker_track_dim: int,
        dt_segment_dim: int,
        csc_segment_dim: int,
        rpc_hit_dim: int,
        gem_hit_dim: int,
        # output dimensions
        output_dim: int,
        # model hyperparameters
        model_dim: int,
        num_heads: int,
        track_latent_len: int,
        muon_det_latent_len: int,
        muon_det_num_processors: int,
        muon_det_processor_block_weight_sharing: bool,
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

        # tracker track: tt
        self.tracker_track_embedder = nn.Linear(in_features=tracker_track_dim, out_features=model_dim)

        # muon detector measurements: mdm
        self.dt_segment_embedder = nn.Linear(in_features=dt_segment_dim, out_features=model_dim)
        self.csc_segment_embedder = nn.Linear(in_features=csc_segment_dim, out_features=model_dim)
        self.rpc_hit_embedder = nn.Linear(in_features=rpc_hit_dim, out_features=model_dim)
        self.gem_hit_embedder = nn.Linear(in_features=gem_hit_dim, out_features=model_dim)

        self.tracker_track_encoder = PerceiverEncoder(
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

        self.muon_det_processor = PerceiverProcessorBlock(
            num_layers=muon_det_num_processors,
            latent_dim=model_dim,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
            weight_sharing=muon_det_processor_block_weight_sharing,
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

        # tracker track-wise regression head
        self.head = nn.Linear(
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
        tracker_track_embed = self.tracker_track_embedder(tracker_track)
        dt_segment_embed = self.dt_segment_embedder(dt_segment)
        csc_segment_embed = self.csc_segment_embedder(csc_segment)
        rpc_hit_embed = self.rpc_hit_embedder(rpc_hit)
        gem_hit_embed = self.gem_hit_embedder(gem_hit)

        # NOTE: muon detector system measurement encoding

        # Combine muon detector measurements
        # embed: (N, L_muon_det, D_model)
        # where L_muon_det = L_dt_seg + L_csc_seg + L_rpc_hit + L_gem_hit
        muon_det_embed = torch.cat(
            tensors=[
                dt_segment_embed,
                csc_segment_embed,
                rpc_hit_embed,
                gem_hit_embed,
            ],
            dim=1, # along sequence length dimension
        )

        muon_det_data_mask = torch.cat(
            tensors=[
                dt_segment_data_mask,
                csc_segment_data_mask,
                rpc_hit_data_mask,
                gem_hit_data_mask,
            ],
            dim=1 # along sequence length dimension
        )

        # NOTE: muonn detector measurement encoding
        muon_det_latent = self.muon_det_encoder(
            input=muon_det_embed,
            data_mask=muon_det_data_mask,
        )

        muon_det_latent = self.muon_det_processor(
            latent=muon_det_latent,
        )

        # NOTE: tracker track encoding
        tracker_track_latent = self.tracker_track_encoder(
            input=tracker_track_embed,
            data_mask=tracker_track_data_mask,
        )

        latent = self.encoder(
            target=tracker_track_latent,
            source=muon_det_latent,
            target_data_mask=None,
            source_data_mask=None,
        )

        tracker_track_embed = self.decoder(
            target=tracker_track_embed,
            source=latent,
            target_data_mask=tracker_track_data_mask,
            source_data_mask=None,
        )

        # NOTE: classification head
        #
        # classification head: (N, L_trk, D_model) -> (N, L_trk, 1)
        tracker_track_embed: Tensor = self.head(tracker_track_embed)
        # squeeze: (N, L_trk, 1) -> (N, L_trk)
        tracker_track_embed = tracker_track_embed.squeeze(dim=2)
        return tracker_track_embed
