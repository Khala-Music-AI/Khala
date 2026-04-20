import lightning as L
import torch
import torch.nn as nn
from dac import Encoder, Decoder
from rvq import ResidualVectorQuantization
from typing import List, Dict, Any, Union


class DacRVQ(L.LightningModule):
    def __init__(self, configs):
        super().__init__()

        self.encoder = Encoder(**configs['encoder'])
        self.decoder = Decoder(**configs['decoder'])

        self.q_dropout = configs['quantizer'].pop('q_dropout')
        self.quantizer = ResidualVectorQuantization(**configs['quantizer'])

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, force_full_quantization: bool = False) -> Union[tuple]:
        # x shape: (batch_size, channels, T)
        assert x.dim() == 3

        z = self.encoder(x)

        if self.training and self.q_dropout and not force_full_quantization:
            # rand_nq = int(torch.randint(1, self.quantizer.num_quantizers + 1, (1,)).item())
            # q_z, codes, commit_loss = self.quantizer(z_t, nq_to_use=rand_nq)

            if torch.rand(()) < 0.2:
                rand_nq = self.quantizer.num_quantizers
            else:
                rand_nq = int(torch.randint(1, self.q_dropout + 1, (1,)).item())
            q_z, codes, commit_loss = self.quantizer(z, nq_to_use=rand_nq)

        else:
            q_z, codes, commit_loss = self.quantizer(z)

        x_pred = self.decoder(q_z)

        return x_pred, q_z, codes, commit_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an input waveform into a list of discrete code tensors."""
        assert x.dim() == 3
        z = self.encoder(x)
        _, codes, _ = self.quantizer(z)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes a list of discrete code tensors back into a waveform."""
        assert isinstance(codes, torch.Tensor), "Input `codes` must be a tensors."
        q_z = self.quantizer.decode(codes)  #q_z shape: (B, C, T)
        x_pred = self.decoder(q_z)
        return x_pred

