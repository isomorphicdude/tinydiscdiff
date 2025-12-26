# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Noise Schedules (SNR).
Ported from: https://github.com/google-deepmind/md4/blob/main/md4/models/diffusion/md4.py
"""
import math
import torch
import torch.nn as nn


class MaskingSchedule:
    """Masking noise schedule."""

    def __init__(self, schedule_fn_type: str = "cosine", eps: float = 1e-4):
        self.schedule_fn_type = schedule_fn_type
        self.eps = eps

    def __call__(self, t):
        # return logSNR(t) = log(alpha(t) / (1 - alpha(t)))
        # final ELBO is in terms of logSNR
        # ∫ₜ₁ᵗ σ(λ) 𝐄_{𝐪 (𝐱_λ | 𝐱_𝐭)} [𝐱₀ᵗ 𝐥𝐨𝐠 μ(𝐱_λ, λ)]
        return torch.log(self.alpha(t) / (1.0 - self.alpha(t)))

    def _dalpha(self, t):
        # here alpha_t = int exp(-beta(s))ds
        if self.schedule_fn_type == "cosine":
            return -math.pi / 2.0 * torch.sin(math.pi / 2.0 * (1.0 - t))
        elif self.schedule_fn_type == "linear":
            return -torch.ones_like(t)
        elif "poly" in self.schedule_fn_type:
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return -exponent * t ** (exponent - 1.0)
        else:
            raise NotImplementedError()

    def dalpha(self, t):
        # time derivative
        return (1.0 - 2 * self.eps) * self._dalpha(t)

    def _alpha(self, t):
        if self.schedule_fn_type == "linear":
            return 1.0 - t
        elif "poly" in self.schedule_fn_type:
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return 1.0 - t**exponent
        elif self.schedule_fn_type == "cosine":
            return 1.0 - torch.cos(math.pi / 2.0 * (1.0 - t))
        else:
            raise NotImplementedError()

    def alpha(self, t: torch.Tensor):
        """Returns alpha(t) same shape as t."""
        # add eps to avoid alpha=0 or 1, so interval [eps, 1-eps]
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps

    def dgamma_times_alpha(self, t: torch.Tensor):
        """Returns alpha'(t) / (1 - alpha(t)) same shape as t."""
        return self.dalpha(t) / (1.0 - self.alpha(t))
