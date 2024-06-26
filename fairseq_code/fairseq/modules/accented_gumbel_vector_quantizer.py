# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class AccentedGumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        input_dim,
        no_entires,
        temp,
        no_accents,
        output_dim,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        hard=True,
        std=0,
    ):
        """Vector quantization using gumbel softmax

        Args:
            input_dim: input dimension (channels)
            num_entires: number of quantized vectors per accent
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            no_accents: number of accents for vector quantization
            output_dim: dimensionality of the resulting quantized vector
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.no_accents = no_accents
        self.input_dim = input_dim
        self.no_entries = no_entires
        self.hard = hard
        self.output_dim = output_dim
        

        # self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        
        self.codebooks = nn.ModuleList([
                nn.Embedding(num_embeddings= self.no_entries, embedding_dim= output_dim)
                for _ in range(no_accents)
            ])
        
        if std == 0:
            for codebook in self.codebooks:
                nn.init.uniform_(codebook.weight)
        else:
            for codebook in self.codebooks:
                nn.init.normal_(codebook.weight, mean=0, std=std)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, self.no_entries),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, self.no_entries)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def forward(self, x, accents, produce_targets=False):
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)

        with torch.no_grad():
            _, k = x.max(-1)
            hard_x = (
                x.new_zeros(*x.shape)
                .scatter_(-1, k.view(-1, 1), 1.0)
            )


        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(
                x
            )
        else:
            x = hard_x

        vars = []
        for label in accents:
            arr = torch.tensor([i for i in range(self.no_entries)], dtype=torch.int).to(x.device)
            vars.append(self.codebooks[label](arr))

        vars = torch.stack(vars) # B x no_entries x output_dim  

        x = torch.matmul(x.view(bsz, tsz, -1), vars)
        return x
