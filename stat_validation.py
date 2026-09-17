#!/usr/bin/env python3
"""
stat_validation.py — Research multiple-testing diagnostics.

Provides:
  - Deflated Sharpe Ratio (DSR)
  - CSCV-style Probability of Backtest Overfitting (PBO)

Important:
  - These diagnostics do NOT promote a model.
  - DSR here treats Phase 2D observations as per-trade returns.
  - Annualization is therefore based on observed trades per year, not
    15-minute bars per year.
  - For sparse/event-driven returns, trade timestamps should be supplied.
  - PBO requires aligned per-observation returns for each historical trial;
    aggregate EV/precision summaries are insufficient.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _clean(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _trade_rate_annualization(
    n_trades: int,
    trade_timestamps: Sequence,
) -> float:
    """Return sqrt(observed trades/year) for event-level returns."""
    if n_trades < 2:
        return 1.0

    ts = pd.to_datetime(
        list(trade_timestamps),
        utc=True,
        errors="coerce",
    ).dropna()

    if len(ts) < 2:
        raise ValueError(
            "At least two valid trade timestamps are required to infer "
            "trades-per-year."
        )

    elapsed_days = (
        float(
            (ts.max() - ts.min()).total_seconds()
        )
        / 86400.0
    )

    if elapsed_days <= 0.0:
        raise ValueError(
            "Trade timestamps must span a positive time interval."
        )

    years = elapsed_days / 365.25

    trades_per_year = n_trades / years

    return sqrt(max(trades_per_year, 1.0))


def sharpe_ratio(
    returns: Iterable[float],
    *,
    trade_timestamps: Sequence | None = None,
    periods_per_year: float | None = None,
) -> float:
    """
    Annualized arithmetic Sharpe.

    For per-trade/event returns:
      - pass trade_timestamps, or
      - pass an independently justified trades-per-year value.

    Do not pass a 15-minute bars/year factor for Phase 2D trade returns.
    """
    r = _clean(returns)

    if len(r) < 2:
        return float("nan")

    sd = float(np.std(r, ddof=1))

    if sd <= 0.0:
        return 0.0

    if periods_per_year is None:
        if trade_timestamps is None:
            raise ValueError(
                "For event-level returns, provide trade_timestamps or "
                "an explicitly justified periods_per_year."
            )

        if len(trade_timestamps) != len(r):
            raise ValueError(
                "trade_timestamps length must match finite returns length."
            )

        annualization = _trade_rate_annualization(
            len(r),
            trade_timestamps,
        )

        # _trade_rate_annualization returns sqrt(trades/year).
        return float(np.mean(r) / sd * annualization)

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    return float(
        np.mean(r) / sd * sqrt(periods_per_year)
    )


def deflated_sharpe_ratio(
    candidate_returns: Iterable[float],
    trial_sharpes: Iterable[float],
    *,
    trade_timestamps: Sequence | None = None,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float | None = None,
    confidence: float = 0.95,
) -> dict:
    """
    Estimate DSR-style survival probability for an event-level return stream.

    candidate_returns:
        Per-trade/event Net-R observations.

    trial_sharpes:
        Sharpe ratios for the full set of historical research trials used
        during selection. They must use the same annualization convention.

    trade_timestamps:
        Timestamps aligned with candidate_returns. Recommended for Phase 2D.
    """
    candidate_raw = np.asarray(
        list(candidate_returns),
        dtype=float,
    )

    finite_mask = np.isfinite(candidate_raw)
    candidate = candidate_raw[finite_mask]

    if trade_timestamps is not None:
        timestamps = np.asarray(
            list(trade_timestamps),
            dtype=object,
        )

        if len(timestamps) != len(candidate_raw):
            raise ValueError(
                "trade_timestamps must align one-to-one with "
                "candidate_returns before finite filtering."
            )

        timestamps = timestamps[finite_mask]
    else:
        timestamps = None

    trials = _clean(trial_sharpes)

    if len(candidate) < 30:
        return {
            "status": "INSUFFICIENT_CANDIDATE_DATA",
            "n_observations": int(len(candidate)),
            "n_trials": int(len(trials)),
        }

    if len(trials) < 2:
        return {
            "status": "INSUFFICIENT_TRIALS",
            "n_observations": int(len(candidate)),
            "n_trials": int(len(trials)),
        }

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    mean_r = float(np.mean(candidate))
    std_r = float(np.std(candidate, ddof=1))

    if std_r <= 0.0:
        return {
            "status": "ZERO_VARIANCE",
            "n_observations": int(len(candidate)),
            "n_trials": int(len(trials)),
        }

    sr_unannualized = mean_r / std_r

    if periods_per_year is None:
        if timestamps is None:
            return {
                "status": "MISSING_TRADE_TIMESTAMPS",
                "message": (
                    "Provide trade timestamps or an explicitly justified "
                    "trades-per-year factor for event-level annualization."
                ),
                "n_observations": int(len(candidate)),
                "n_trials": int(len(trials)),
            }

        annualization = _trade_rate_annualization(
            len(candidate),
            timestamps,
        )
        trades_per_year = annualization ** 2
    else:
        if periods_per_year <= 0.0:
            raise ValueError(
                "periods_per_year must be positive."
            )

        annualization = sqrt(periods_per_year)
        trades_per_year = periods_per_year

    candidate_sharpe = (
        sr_unannualized
        * annualization
    )

    skew = float(
        stats.skew(
            candidate,
            bias=False,
        )
    )

    kurtosis_pearson = float(
        stats.kurtosis(
            candidate,
            fisher=False,
            bias=False,
        )
    )

    if not np.isfinite(skew):
        skew = 0.0

    if not np.isfinite(kurtosis_pearson):
        kurtosis_pearson = 3.0

    n_trials = len(trials)

    trial_mean = float(
        np.mean(trials)
    )

    trial_std = float(
        np.std(
            trials,
            ddof=1,
        )
    )

    gamma = 0.5772156649015329

    if trial_std <= 0.0:
        expected_max = (
            benchmark_sharpe
            + trial_mean
        )
    else:
        p1 = 1.0 - 1.0 / n_trials
        p2 = 1.0 - 1.0 / (
            n_trials * np.e
        )

        expected_z = (
            (1.0 - gamma)
            * stats.norm.ppf(p1)
            + gamma
            * stats.norm.ppf(p2)
        )

        expected_max = (
            benchmark_sharpe
            + trial_std
            * expected_z
        )

    # Non-normality-adjusted standard error. The annualization factor is
    # based on observed event frequency, not bar frequency.
    sr_se = sqrt(
        max(
            0.0,
            (
                1.0
                - skew * sr_unannualized
                + (
                    (kurtosis_pearson - 1.0)
                    / 4.0
                )
                * sr_unannualized**2
            )
            / len(candidate),
        )
    ) * annualization

    if sr_se <= 0.0:
        return {
            "status": "ZERO_SHARPE_STANDARD_ERROR",
            "candidate_sharpe": round(
                candidate_sharpe,
                6,
            ),
            "expected_max_null_sharpe": round(
                expected_max,
                6,
            ),
            "n_observations": int(
                len(candidate)
            ),
            "n_trials": int(
                n_trials
            ),
        }

    z = (
        candidate_sharpe
        - expected_max
    ) / sr_se

    survival_probability = float(
        stats.norm.cdf(z)
    )

    return {
        "status": "OK",
        "candidate_sharpe": round(
            candidate_sharpe,
            6,
        ),
        "expected_max_null_sharpe": round(
            expected_max,
            6,
        ),
        "n_observations": int(
            len(candidate)
        ),
        "n_trials": int(
            n_trials
        ),
        "trades_per_year_used": round(
            float(trades_per_year),
            6,
        ),
        "skewness": round(
            skew,
            6,
        ),
        "kurtosis_pearson": round(
            kurtosis_pearson,
            6,
        ),
        "dsr_z": round(
            z,
            6,
        ),
        "dsr_survival_probability": round(
            survival_probability,
            6,
        ),
        "p_value": round(
            1.0 - survival_probability,
            6,
        ),
        "confidence_gate": float(
            confidence
        ),
        "passes_confidence_gate": bool(
            survival_probability >= confidence
        ),
    }


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    *,
    n_splits: int = 8,
    purge_bars: int = 0,
) -> dict:
    """
    CSCV-style PBO estimate.

    returns_matrix:
        Chronologically aligned per-observation returns.
        Columns represent historical trials.

    purge_bars:
        Generic row-based purge. This is NOT event-aware purging.
        For overlapping 24-bar labels, a true event-aware purge requires
        event start/end timestamps and should be implemented at the caller.
    """
    if not isinstance(
        returns_matrix,
        pd.DataFrame,
    ):
        raise TypeError(
            "returns_matrix must be a pandas DataFrame."
        )

    if returns_matrix.shape[1] < 2:
        return {
            "status": "NEED_AT_LEAST_TWO_TRIALS",
            "pbo": None,
        }

    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError(
            "n_splits must be an even integer >= 4."
        )

    if purge_bars < 0:
        raise ValueError(
            "purge_bars cannot be negative."
        )

    x = (
        returns_matrix
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(how="any")
    )

    t = len(x)

    if t < n_splits * 4:
        return {
            "status": "INSUFFICIENT_ROWS_FOR_CSCV",
            "rows": int(t),
            "n_splits": int(n_splits),
            "pbo": None,
        }

    blocks = np.array_split(
        np.arange(t),
        n_splits,
    )

    half = n_splits // 2

    partitions = list(
        combinations(
            range(n_splits),
            half,
        )
    )

    oos_ranks = []

    for is_blocks in partitions:
        oos_blocks = [
            b
            for b in range(n_splits)
            if b not in is_blocks
        ]

        is_idx = np.concatenate(
            [
                blocks[b]
                for b in is_blocks
            ]
        )

        oos_idx = np.concatenate(
            [
                blocks[b]
                for b in oos_blocks
            ]
        )

        if purge_bars:
            oos_set = set(
                oos_idx.tolist()
            )

            is_idx = np.asarray(
                [
                    i
                    for i in is_idx
                    if not any(
                        abs(i - o)
                        <= purge_bars
                        for o in oos_set
                    )
                ],
                dtype=int,
            )

        if (
            len(is_idx) < 2
            or len(oos_idx) < 2
        ):
            continue

        is_ret = x.iloc[is_idx]
        oos_ret = x.iloc[oos_idx]

        is_sd = (
            is_ret.std(
                ddof=1
            )
            .replace(
                0.0,
                np.nan,
            )
        )

        oos_sd = (
            oos_ret.std(
                ddof=1
            )
            .replace(
                0.0,
                np.nan,
            )
        )

        is_sr = (
            is_ret.mean()
            / is_sd
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(-np.inf)

        oos_sr = (
            oos_ret.mean()
            / oos_sd
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(-np.inf)

        winner = is_sr.idxmax()

        rank = (
            oos_sr.rank(
                pct=True,
                method="average",
            )
            .get(
                winner,
                np.nan,
            )
        )

        if np.isfinite(rank):
            oos_ranks.append(
                float(rank)
            )

    if not oos_ranks:
        return {
            "status": "NO_VALID_CSCV_COMBINATIONS",
            "pbo": None,
            "cscv_combinations": 0,
        }

    pbo = float(
        np.mean(
            [
                rank < 0.50
                for rank in oos_ranks
            ]
        )
    )

    return {
        "status": "OK",
        "pbo": round(
            pbo,
            6,
        ),
        "mean_oos_rank": round(
            float(
                np.mean(oos_ranks)
            ),
            6,
        ),
        "median_oos_rank": round(
            float(
                np.median(oos_ranks)
            ),
            6,
        ),
        "cscv_combinations": len(
            oos_ranks
        ),
        "n_trials": int(
            x.shape[1]
        ),
        "n_rows": int(
            x.shape[0]
        ),
        "purge_bars": int(
            purge_bars
        ),
        "passes_pbo_gate": bool(
            pbo <= 0.25
        ),
    }


if __name__ == "__main__":
    print(
        "stat_validation.py loaded successfully; "
        "no model promotion decision performed."
    )
