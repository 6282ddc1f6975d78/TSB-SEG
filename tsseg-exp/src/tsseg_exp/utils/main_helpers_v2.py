"""Helpers v2 — factored utilities shared across pipeline files.

Backward-compatible with ``main_helpers``; this module adds two
convenience functions that were previously duplicated across pipelines:

* ``normalize_supervision_mode`` — canonical normalisation in one place.
* ``resolve_experiment_cfg``    — resolve hierarchical / flattened config.
* ``per_trial_deadline``        — per-trial time budget from a global budget.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from omegaconf import DictConfig

# Re-export everything from the original module so that
# ``from main_helpers_v2 import ...`` is a drop-in replacement.
from tsseg_exp.utils.main_helpers import *  # noqa: F401, F403
from tsseg_exp.utils.main_helpers import (  # explicit for type-checkers
    _as_monotonic_deadline,
    _as_positive_float,
    is_deadline_exceeded,
    resolve_parent_deadline,
)


# ── Normalisation helpers ─────────────────────────────────────────────


def normalize_supervision_mode(raw: Any) -> str:
    """Return a canonical supervision mode string.

    Accepted values are ``"unsupervised"``, ``"semi_supervised"``, and
    ``"supervised"``.  Anything else (including ``None``) falls back to
    ``"unsupervised"``.
    """
    if raw is None:
        return "unsupervised"
    normalised = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if normalised in {"semi_supervised", "supervised"}:
        return normalised
    return "unsupervised"


def resolve_experiment_cfg(cfg: DictConfig) -> DictConfig:
    """Return the experiment sub-config regardless of nesting style.

    Supports both ``cfg.experiment.*`` and flattened ``cfg.*`` layouts
    produced by ``@package _global_``.
    """
    if "experiment" in cfg:
        return cfg.experiment
    return cfg


# ── Per-trial deadline budget ─────────────────────────────────────────


def per_trial_deadline(
    *,
    budget_seconds: Optional[float],
    n_trials: int,
    elapsed: float = 0.0,
    min_seconds: float = 30.0,
    weight: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute a per-trial budget from a global envelope.

    Parameters
    ----------
    budget_seconds : Optional[float]
        Total seconds allocated for the full run.  ``None`` disables the
        time constraint entirely.
    n_trials : int
        Number of remaining trials (must be >= 1).
    elapsed : float
        Seconds already consumed since the run started.
    min_seconds : float
        Minimum per-trial budget (prevents absurdly short deadlines).
    weight : Optional[float]
        Fraction of the remaining budget to allocate to this trial
        (value in ``(0, 1]``).  When ``None``, the remaining budget is
        divided equally among ``n_trials``.

        Typical usage: ``weight = trial_length / sum_of_remaining_lengths``
        so that longer time series receive proportionally more time.

    Returns
    -------
    trial_budget_seconds : Optional[float]
        Per-trial budget, or ``None`` when unconstrained.
    trial_deadline : Optional[float]
        Absolute monotonic deadline for the upcoming trial.
    """
    if budget_seconds is None:
        return None, None

    remaining = max(budget_seconds - elapsed, 0.0)
    if weight is not None and 0.0 < weight <= 1.0:
        per_trial = remaining * weight
    else:
        n = max(n_trials, 1)
        per_trial = remaining / n
    per_trial = max(per_trial, min_seconds)
    # Cap so that a single trial cannot eat more than the remaining budget.
    per_trial = min(per_trial, remaining)
    return per_trial, _as_monotonic_deadline(per_trial)
