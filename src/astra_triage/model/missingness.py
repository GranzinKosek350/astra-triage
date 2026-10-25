"""The two missingness mechanisms: a stale-stream token and a structural absence token.

Ref: Sec. 2.3.0.3, "The missingness embedding that is learned is added to the
newest observation of a stream for which the last observation is older than 24
hours, indicating to the model that this state of the stream is unknown rather than
unchanged."

The two tokens are deliberately separate objects. A stale stream was measured and
has since gone quiet, which is evidence about the ward's attention. A structurally
absent stream is never measured in that resource at all, as with the ultrasound
branch in the external cohort, so the token says that nothing about the stream can
be inferred rather than that the stream is quiet.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.endpoints.definition import STALE_AFTER_HOURS
from astra_triage.model.config import AstraConfig


class MissingnessTokens(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.stale: nn.Parameter = nn.Parameter(torch.zeros(config.d_model))
        self.absent: nn.Parameter = nn.Parameter(torch.zeros(config.d_model))
        self.enabled: bool = config.use_missingness_token

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        tau_hours: torch.Tensor,
        index: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the embeddings with the stale token applied, and the per-case flag."""
        newest = torch.where(mask, tau_hours, torch.full_like(tau_hours, -1e9)).max(dim=-1).values
        stale = present & ((index - newest) > STALE_AFTER_HOURS)
        if not self.enabled:
            return embeddings, stale
        newest_index = torch.argmax(torch.where(mask, tau_hours, torch.full_like(tau_hours, -1e9)), dim=-1)
        selector = torch.zeros_like(mask)
        selector.scatter_(-1, newest_index.unsqueeze(-1), True)
        selector = selector & stale.unsqueeze(-1)
        token = self.stale.view(1, 1, -1) * selector.unsqueeze(-1).to(embeddings.dtype)
        return embeddings + token, stale

    def absent_embedding(self, reference: torch.Tensor, width: int) -> torch.Tensor:
        return self.absent.view(1, 1, -1).expand(reference.shape[0], width, -1)

    def absent_flag(self, reference: torch.Tensor, width: int) -> torch.Tensor:
        """A one-hot mask that keeps exactly one slot alive for an absent stream."""
        mask = torch.zeros((reference.shape[0], width), dtype=torch.bool, device=reference.device)
        mask[:, 0] = True
        return mask
