# SPDX-License-Identifier: Apache-2.0
"""Synchronous survival policy for validating eager DSpark verification.

This is an opt-in diagnostic policy, not a replacement for upstream cost-based
adaptive verification. Keep budget selection separate from runner/GDN layout.
"""

import math
from collections.abc import Mapping

import numpy as np


def validate_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("DSpark eager survival threshold must be finite and in [0, 1]")
    return value


def select_capacities(confidence: np.ndarray, scheduled: np.ndarray, threshold: float, max_drafts: int) -> np.ndarray:
    """Retain prefixes above threshold, then respect the sampler logits limit.

    Stable global ordering resolves tied survival scores in prefix order. This
    prevents capacity overflow without allowing later positions to displace an
    earlier position with the same score. Missing confidence is handled by the
    caller; malformed confidence is an error, never an implicit trim decision.
    """
    threshold = validate_threshold(threshold)
    confidence = np.asarray(confidence, dtype=np.float32)
    scheduled = np.asarray(scheduled, dtype=np.int32)
    if confidence.ndim != 2 or scheduled.shape != (confidence.shape[0],):
        raise ValueError("confidence must be [B,K] and scheduled must be [B]")
    if np.any(scheduled < 0) or np.any(scheduled > confidence.shape[1]) or max_drafts < 0:
        raise ValueError("invalid scheduled draft count or draft budget")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and in [0,1]")
    survival = np.cumprod(confidence, axis=1, dtype=np.float64)
    eligible = (np.arange(confidence.shape[1])[None, :] < scheduled[:, None]) & (survival >= threshold)
    capacities = eligible.sum(axis=1, dtype=np.int32)
    if int(capacities.sum()) > max_drafts:
        scores = np.where(eligible, survival, -np.inf).ravel()
        winners = np.argsort(-scores, kind="stable")[:max_drafts]
        capacities = np.bincount(winners // confidence.shape[1], minlength=len(scheduled)).astype(np.int32)
    return capacities


def batch_layout(req_ids: list[str], capacities: Mapping[str, int], non_drafts: Mapping[str, int], bonus: int):
    """Exact CPU boundaries in the runner's current request order."""
    caps = np.array([capacities[r] for r in req_ids], dtype=np.int32)
    queries = caps + np.array([non_drafts[r] for r in req_ids], dtype=np.int32)
    qsl = np.zeros(len(req_ids) + 1, dtype=np.int32)
    logits = np.zeros_like(qsl)
    np.cumsum(queries, out=qsl[1:])
    np.cumsum(caps + bonus, out=logits[1:])
    return caps, queries, qsl, logits
