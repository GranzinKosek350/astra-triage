"""Site adapter: reporting practice without a site label at inference.

Ref: Sec. 2.3.0.3, "A site adapter is used to account for differences in reporting
practice across sites without requiring additional information at inference except
for the site of origin."

The adapter is a rank-limited residual map per site, so a site can shift and rotate
the stream embeddings it produces while leaving the shared geometry intact. It is
initialised at zero, which means the untrained network is site-agnostic and the
adapter has to earn its effect. The adversarial term in the objective is what stops
the adapter from becoming a site classifier.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.model.config import AstraConfig


class SiteAdapter(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.rank: int = config.adapter_rank
        self.down: nn.Linear = nn.Linear(config.d_model, config.adapter_rank, bias=False)
        self.up: nn.Parameter = nn.Parameter(torch.zeros(config.site_count, config.d_model, config.adapter_rank))

    def forward(self, embeddings: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        low = self.down(embeddings)
        per_site = self.up[site]
        adapted: torch.Tensor = embeddings + torch.einsum("bnr,bdr->bnd", low, per_site)
        return adapted
