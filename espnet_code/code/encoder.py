# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
import logging
from typing import Optional, Tuple

import torch
from typeguard import check_argument_types

from abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import fairseq

class HubertEncoder(AbsEncoder):
    """FairSeq Hubert encoder module, used for loading pretrained weight and finetuning

    Args:
        hubert_url: url to Hubert pretrained model
        hubert_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        output_size: dimension of attention
        normalize_before: whether to use layer_norm before the first block
        freeze_finetune_updates: steps that freeze all layers except output layer
            before tuning the whole model (nessasary to prevent overfit).
        dropout_rate: dropout rate
        activation_dropout: dropout rate in activation function
        attention_dropout: dropout rate in attention
    Hubert specific Args:
        Please refer to:
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert.py
    """

    def __init__(
        self,
        model_file_name: str,
        hubert_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str = "static",
        layerdrop: float = 0.1,
        feature_grad_mult: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()
        self.apply_mask = apply_mask
        try:
            import fairseq
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        arg_overrides = {
            "dropout": dropout_rate,
            "activation_dropout": activation_dropout,
            "attention_dropout": attention_dropout,
            "mask_length": mask_length,
            "mask_prob": mask_prob,
            "mask_selection": mask_selection,
            "mask_other": mask_other,
            "mask_channel_length": mask_channel_length,
            "mask_channel_prob": mask_channel_prob,
            "mask_channel_selection": mask_channel_selection,
            "mask_channel_other": mask_channel_other,
            "encoder_layerdrop": layerdrop,
            "feature_grad_mult": feature_grad_mult,
            "data": hubert_dir_path,
        }

        self.hubert_model_path = f'{hubert_dir_path}/{model_file_name}'

        (
            models,
            self.pretrained_cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.hubert_model_path],
            arg_overrides=arg_overrides,
            strict=False,
        )
        model = models[0]

        d = self.pretrained_cfg.model.encoder_embed_dim
        self.pretrained_params = copy.deepcopy(model.state_dict())

        self._output_size = output_size

        self.encoders = model
        
        
        # Freeze feature extractor
        for param in self.encoders.feature_extractor.parameters():
            param.requires_grad = False
        
        if self.encoders.post_extract_proj is not None:
            for param in self.encoders.post_extract_proj.parameters():
                param.requires_grad = False
        
        for param in self.encoders.encoder.pos_conv.parameters():
            param.requires_grad = False
    
        # Freeze first 5 layers of Encoder
        for layer in self.encoders.encoder.layers[:5]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Freeze codebook weights
        for codebook in self.encoders.codebooks:
            for param in codebook.parameters():
                param.requires_grad = False

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if output_size and output_size != d:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(d, output_size),
            )
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        accent_labels: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert ASR Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                padding_mask=masks,
                mask=self.apply_mask and self.training,
                features_only=True,
                output_layer=None,
                accents=accent_labels
            )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)

        # save gpu memory
        del enc_outputs

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        
        masks = masks.unsqueeze(1)
        masks = (~masks)
        return xs_pad, masks, None, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")