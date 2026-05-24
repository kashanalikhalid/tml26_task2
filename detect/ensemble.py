"""Combine per-suspect features into a single stealing-confidence score.

For TPR@5%FPR the ranking is all that matters, not the absolute values. We use
a rank-mean (Borda) ensemble over a small set of features so individual feature
outliers (e.g. a single suspect with exact weight match) don't dominate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# +1 means "higher feature value = more likely stolen". -1 inverts that.
DEFAULT_FEATURE_SIGNS: dict[str, int] = {
    # Behavioral
    "behavioral_jsd": -1,
    "behavioral_sym_kl": -1,
    "behavioral_ce_to_target": -1,
    "behavioral_top1_agree": +1,
    "behavioral_top5_member": +1,
    "behavioral_logit_cosine": +1,
    "behavioral_logit_pearson": +1,
    # Weight-space
    "weight_cosine_full": +1,
    "weight_cosine_backbone": +1,
    "weight_exact_tensor_frac": +1,
    "weight_l2_relative": -1,
}


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to ranks in [0, 1]. Higher value -> higher rank."""
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = values.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def ensemble_score(
    features: pd.DataFrame,
    feature_signs: dict[str, int] | None = None,
) -> np.ndarray:
    feature_signs = feature_signs or DEFAULT_FEATURE_SIGNS
    used = [c for c in feature_signs if c in features.columns]
    if not used:
        raise ValueError("No known feature columns found in DataFrame.")

    ranks = np.zeros(len(features), dtype=np.float64)
    for col in used:
        values = features[col].to_numpy(dtype=np.float64) * float(feature_signs[col])
        ranks += rank_normalize(values)
    return ranks / len(used)
