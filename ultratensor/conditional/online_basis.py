"""ONB — online basis via Oja's rule (port of online_basis.c).

When speculative decoding rejects a draft, the residual
``target_hidden - draft_hidden`` carries exactly the directions the
compressed path got wrong. ONB feeds those residuals into a deflated
Oja update so each layer's PCA basis tracks the live error manifold —
the closed feedback loop any drafter we build needs.

Semantics are copied from the C: queue cap 256, update gate of 4
rejections, eta decay ``eta0 / sqrt(t + 1)``, Gram-Schmidt-style
deflation so components do not collapse onto PC1, identity start when no
existing basis is supplied, and ``reproject_weight`` producing
W_proj[m x k] = W_orig[m x dim] @ basis^T[dim x k].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ONB_QUEUE_CAP = 256


@dataclass
class _LayerState:
    dim: int
    k: int
    eta0: float
    W: np.ndarray  # [k, dim] row-major basis
    t: int = 0
    version: int = 0
    active: bool = False

    def oja_update(self, x: np.ndarray) -> None:
        """One deflated Oja sweep over all k components."""
        eta = self.eta0 / np.sqrt(self.t + 1)
        xd = x.astype(np.float64, copy=True)
        for i in range(self.k):
            wi = self.W[i]  # view; mutate in place below
            score = float(xd @ wi)
            wi += eta * score * xd
            norm = np.linalg.norm(wi)
            if norm > 1e-12:
                wi /= norm
            # deflate: remove the new component's projection from xd
            proj = float(xd @ wi)
            xd -= proj * wi
        self.t += 1


class OnlineBasis:
    """Port of onb_ctx_t with the same defaults and gates."""

    def __init__(
        self,
        dims,
        ks,
        existing_bases=None,
        eta0: float = 0.01,
        updates_per_rejection: int = 1,
        min_rejections_before_update: int = 4,
        queue_cap: int = ONB_QUEUE_CAP,
    ) -> None:
        dims = [int(d) for d in dims]
        ks = [int(k) for k in ks]
        if not dims or len(dims) != len(ks):
            raise ValueError("dims and ks must be non-empty and equal length")
        self.n_layers = len(dims)
        self.eta0 = eta0 if eta0 > 0.0 else 0.01
        self.updates_per_rejection = updates_per_rejection
        self.min_rejections_before_update = min_rejections_before_update
        self.queue_cap = max(1, queue_cap)
        self.queue: list[tuple[int, np.ndarray]] = []
        self.total_rejections = 0
        self.total_updates = 0
        self.basis_version = [0] * self.n_layers

        self.layers: list[_LayerState] = []
        for l, (dim, k) in enumerate(zip(dims, ks)):
            k = min(max(1, k), dim)
            W = np.zeros((k, dim), dtype=np.float64)
            if existing_bases is not None and existing_bases[l] is not None:
                base = np.asarray(existing_bases[l], dtype=np.float64)
                W[:] = base[:k, :dim]
            else:
                # identity start: w_i = e_i
                for i in range(k):
                    W[i, i] = 1.0
            self.layers.append(_LayerState(dim=dim, k=k, eta0=self.eta0, W=W))

    def _push(self, layer: int, residual: np.ndarray) -> None:
        if len(self.queue) >= self.queue_cap:
            self.queue = self.queue[1:] + [(layer, residual)]
        else:
            self.queue.append((layer, residual))
        self.total_rejections += 1

    def record_rejection(self, layer: int, target_hidden, draft_hidden) -> int:
        """Enqueue the residual of a rejected draft at ``layer``."""
        self._check_layer(layer)
        residual = (
            np.asarray(target_hidden, dtype=np.float64)
            - np.asarray(draft_hidden, dtype=np.float64)
        )[: self.layers[layer].dim]
        self._push(layer, residual)
        return 0

    def record_residual(self, layer: int, residual) -> int:
        self._check_layer(layer)
        self._push(
            layer, np.asarray(residual, dtype=np.float64)[: self.layers[layer].dim]
        )
        return 0

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range [0, {self.n_layers})")

    def apply_pending(self) -> int:
        """Apply queued residuals (gated like the C onb_apply_pending)."""
        if not self.queue or self.total_rejections < self.min_rejections_before_update:
            return 0
        updates = 0
        for layer, residual in self.queue:
            ls = self.layers[layer]
            norm = np.linalg.norm(residual)
            if norm < 1e-12:
                continue
            x_norm = residual / norm
            for _ in range(self.updates_per_rejection):
                ls.oja_update(x_norm)
            ls.active = True
            ls.version += 1
            self.basis_version[layer] = ls.version
            self.total_updates += 1
            updates += 1
        self.queue = []
        return updates

    def reproject_weight(self, W_orig: np.ndarray, layer: int) -> np.ndarray:
        """W_proj[m x k] = W_orig[m x dim] @ basis^T[dim x k]."""
        self._check_layer(layer)
        ls = self.layers[layer]
        return W_orig @ ls.W.T

    def coverage(self, vector, layer: int) -> float:
        """Fraction of the vector's energy captured by the current basis.

        This is the cheap reconstruction-risk readout for G10: a low
        coverage on a draft residual means the compressed path is
        missing directions the live traffic actually uses -> escalate.
        """
        self._check_layer(layer)
        v = np.asarray(vector, dtype=np.float64)
        n = float(v @ v)
        if n < 1e-12:
            return 1.0
        ls = self.layers[layer]
        proj = ls.W @ v
        return float((proj @ proj) / n)
