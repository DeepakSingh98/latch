"""FGM expert-choice implementation with the common FP32 SiLU activation.

Modified adaptation (2026-10-08) of research/fgm_a/model.py::ExpertChoice at
FGM-D 5e6403a91af85d81d8d06239c866f2e6407c4515. Apache-2.0 llm-random attribution
and license are preserved in research/fgm_a/UPSTREAM.md and UPSTREAM_LICENSE.txt.
The upstream source is pinned at 810e6451a402f9a37c2f6f43c1bb75bc00f37140.

Grouping, initialization, routing and statistics are inherited unchanged. The
copied donor forward differs only in its non-gated expert activation: SiLU in
FP32 (FP64 audit supported) replaces BF16 ReLU to match the common dense FFN.
"""
import torch
from torch.nn import functional as F

from .fgm_model import ExpertChoice as DonorExpertChoice


class ExpertChoice(DonorExpertChoice):
    def forward(self, x):
        original_shape = x.shape
        grouped = self._group(x)
        positions = grouped.shape[1]
        weights, indices = self.route(x)
        flat_indices = indices * positions + torch.arange(positions, device=x.device)
        extracted = grouped.reshape(-1, self.config.d_model).index_select(
            0, flat_indices.reshape(-1)).reshape(self.n_experts, -1, self.config.d_model)
        # Only adaptation to the donor forward: common non-gated FP32 SiLU.
        projected = torch.bmm(extracted, self.w1).to(self.w1.dtype)
        expert_output = torch.bmm(F.silu(projected), self.w2)
        # Original selected probabilities weight outputs; never renormalize
        # them over selected experts. FP32 reduction avoids BF16 atomics.
        weighted = expert_output.to(weights.dtype) * weights.reshape(self.n_experts, -1, 1)
        merged = torch.zeros((grouped.shape[0] * positions, self.config.d_model),
                             dtype=weighted.dtype, device=x.device)
        merged = merged.index_add(0, flat_indices.reshape(-1), weighted.reshape(-1, self.config.d_model))
        if self.capture_routing:
            with torch.no_grad():
                counts = torch.bincount(flat_indices.reshape(-1), minlength=grouped.shape[0] * positions)
                self._routing_snapshot = {
                    "selection_histogram": torch.bincount(counts, minlength=self.n_experts + 1),
                    "expert_selections": torch.full((self.n_experts,), self.capacity * positions,
                                                    dtype=torch.int64, device=x.device),
                    "selected_weight_mean": weights.detach().mean(),
                    "selected_weight_min": weights.detach().min(),
                    "selected_weight_max": weights.detach().max(),
                }
        # Retain the affine bias for tokens selected by no expert, as upstream.
        return self.output_norm(merged.reshape(original_shape))
