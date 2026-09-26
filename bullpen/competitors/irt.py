"""Item response theory: the 2PL and its multidimensional generalisation.

``P(model m gets question q right) = sigmoid(a_q . theta_m + b_q)``, fitted by joint MLE;
``dim = 1`` is the classic 2PL (``bits_irt2pl``). A held-out model's ``theta`` is the
penalised MLE with the item parameters frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from bullpen.models.base import Encoder, capped_threads, torch_device

#: Full-batch Adam settings for the joint MLE.
FIT_STEPS: int = 1500
FIT_LR: float = 0.05

#: L2 on theta and the discriminations (mean square). Pins the theta/a scale and keeps
#: a model that answers every probe correctly at a finite fold-in ability.
FIT_L2: float = 1e-3

#: Newton's method on the frozen-item fold-in (concave in theta).
FOLDIN_MAX_STEPS: int = 50
FOLDIN_TOL: float = 1e-8

#: Damping added to the Hessian before the Newton solve (covers ``l2 = 0``).
FOLDIN_DAMPING: float = 1e-6

#: Logit clamp inside the sigmoid, to keep ``exp`` from overflowing.
_LOGIT_CLIP: float = 60.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, clipped so an extreme logit saturates instead of overflowing."""
    z = np.clip(np.asarray(z, dtype=np.float64), -_LOGIT_CLIP, _LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class IRTEncoder(Encoder):
    """(Multidimensional) 2PL fitted by joint MLE on soft per-cell accuracy targets."""

    steps: int = FIT_STEPS
    lr: float = FIT_LR
    l2: float = FIT_L2
    #: [Q, dim] item discriminations
    a: np.ndarray | None = field(default=None, repr=False)
    #: [Q] item easinesses; the sign convention is ``+b``, so larger is easier
    b: np.ndarray | None = field(default=None, repr=False)

    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> IRTEncoder:
        import torch

        if self.steps < 1:
            raise ValueError(f"{self.name}: steps must be at least 1, got {self.steps}")
        A = self._prepare(A, cols)
        device = torch_device()
        Y = torch.tensor(A.astype(np.float32), device=device)
        n_models, n_questions = Y.shape

        # drawn on CPU so the initialisation does not depend on the device
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        theta = (0.1 * torch.randn(n_models, self.dim, generator=gen)).to(device)
        theta.requires_grad_(True)
        # centred on 1.0: questions start out positively discriminating
        a = (0.1 * torch.randn(n_questions, self.dim, generator=gen) + 1.0).to(device)
        a.requires_grad_(True)
        b = torch.zeros(n_questions, device=device, requires_grad=True)

        opt = torch.optim.Adam([theta, a, b], lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.steps):
            opt.zero_grad()
            loss = loss_fn(theta @ a.T + b, Y) + self.l2 * (theta.pow(2).mean() + a.pow(2).mean())
            loss.backward()
            opt.step()

        self.X = theta.detach().cpu().numpy().astype(np.float64)
        self.a = a.detach().cpu().numpy().astype(np.float64)
        self.b = b.detach().cpu().numpy().astype(np.float64)
        logger.debug(
            f"{self.name}: {n_models} models x {n_questions} questions, dim {self.dim}, "
            f"{self.steps} steps, final penalised BCE {float(loss):.4f}"
        )
        return self

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Penalised MLE of theta for one model with the items frozen (Newton from zero)."""
        j = self.local(cols)
        a, b = self.a[j], self.b[j]
        y = np.asarray(bits, dtype=np.float64).ravel()
        if y.shape[0] != j.shape[0]:
            raise ValueError(f"{self.name}: {j.shape[0]} columns but {y.shape[0]} bits")
        eye = np.eye(self.dim)
        theta = np.zeros(self.dim)
        for _ in range(FOLDIN_MAX_STEPS):
            p = sigmoid(a @ theta + b)
            grad = a.T @ (y - p) - self.l2 * theta
            hess = -(a * (p * (1.0 - p))[:, None]).T @ a - self.l2 * eye
            step = np.linalg.solve(hess - FOLDIN_DAMPING * eye, grad)
            theta = theta - step
            if np.abs(step).max() < FOLDIN_TOL:
                break
        else:
            logger.debug(
                f"{self.name}: fold-in did not converge in {FOLDIN_MAX_STEPS} Newton steps"
            )
        return theta

    def predict(self, theta: np.ndarray, cols: np.ndarray) -> np.ndarray:
        j = self.local(cols)
        return sigmoid(self.a[j] @ np.asarray(theta, dtype=np.float64) + self.b[j])


__all__ = ["IRTEncoder", "sigmoid"]
