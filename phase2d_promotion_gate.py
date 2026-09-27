"""Phase 2D prospective statistical promotion gate.

Primary decision method:
  - UTC calendar-day blocks keyed by prediction open_time.
  - One daily mean Net-R per model.
  - Candidate and production paired on common UTC days.
  - Paired block bootstrap of the daily-mean series.
  - Four fixed looks: 15 / 20 / 25 / 30 paired blocks.
  - Equal alpha spending: 0.0125 per look (overall alpha <= 0.05
    by Bonferroni/union-bound control across the four looks).
  - 98.75% two-sided confidence intervals at each look.
  - Checkpoint decisions are persisted to experiment_ledger.jsonl.
  - Once a checkpoint is evaluated, that look is never re-evaluated.
  - Relative early harm is a checkpoint decision: upper CI of
    candidate-minus-production < 0.
  - Promotion requires both candidate profitability and superiority
    over production, plus coverage gates.

Regime coverage intentionally fails closed until a frozen
phase2d_regime_reference.json exists. No training-era cutoffs are
invented here.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DAY_MS = 24 * 60 * 60 * 1000
BLOCK_MS = DAY_MS

MIN_RESOLVED = 50

# Backwards-compatible import name used by the validator.
MIN_UNIQUE_BLOCKS = 15
MIN_INDIVIDUAL_BLOCKS = 15
MIN_PAIRED_BLOCKS = 15

CHECKPOINT_BLOCKS = (15, 20, 25, 30)
FINAL_CHECKPOINT = 30

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 42

# Four equally allocated looks: 0.05 / 4.
ALPHA_TOTAL = 0.05
ALPHA_PER_CHECKPOINT = ALPHA_TOTAL / len(CHECKPOINT_BLOCKS)
CONF = 1.0 - ALPHA_PER_CHECKPOINT

MAX_CALENDAR_DAYS = 90

AUDIT_LEDGER_FILE = Path(
    "research_outputs/experiment_ledger.jsonl"
)

STATE_FILE = Path(
    "research_outputs/phase2d_state.json"
)

REGIME_REFERENCE_FILE = Path(
    "phase2d_regime_reference.json"
)

CHECKPOINT_EVENT = "F08_CHECKPOINT_EVALUATED"
HARM_EVENT = "F08_EARLY_HARM_STOP"


def _clean(records: list[dict]) -> pd.DataFrame:
    """Keep only economically resolved observations."""

    rows = [
        r
        for r in records
        if r.get("status") in {"TP", "SL", "EXPIRED"}
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "resolved_at",
                "net_r",
                "status",
                "id",
                "signal",
                "regime_atr_pct",
            ]
        )

    for column in ("open_time", "net_r"):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "id" not in df.columns:
        df["id"] = ""

    if "signal" not in df.columns:
        df["signal"] = ""

    if "resolved_at" not in df.columns:
        df["resolved_at"] = None

    if "regime_atr_pct" not in df.columns:
        df["regime_atr_pct"] = np.nan

    df = df[
        np.isfinite(df["open_time"])
        & np.isfinite(df["net_r"])
    ].copy()

    return df.reset_index(drop=True)


def _day_id_from_open_time(open_time: pd.Series) -> pd.Series:
    return (
        open_time.astype("int64")
        // BLOCK_MS
    )


def _daily_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse clean observations to one mean Net-R per UTC day."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "day_id",
                "daily_mean_net_r",
                "n_obs",
                "mean_atr_pct",
            ]
        )

    x = df.copy()
    x["day_id"] = _day_id_from_open_time(
        x["open_time"]
    )

    x["regime_atr_pct"] = pd.to_numeric(
        x["regime_atr_pct"],
        errors="coerce",
    )

    grouped = (
        x.groupby("day_id", sort=True)
        .agg(
            daily_mean_net_r=("net_r", "mean"),
            n_obs=("net_r", "size"),
            mean_atr_pct=("regime_atr_pct", "mean"),
        )
        .reset_index()
    )

    return grouped


def _unique_block_count(df: pd.DataFrame) -> int:
    return int(
        len(
            _daily_means(df)
        )
    )


def _paired_daily_means(
    candidate: pd.DataFrame,
    production: pd.DataFrame,
) -> pd.DataFrame:
    c = _daily_means(candidate).rename(
        columns={
            "daily_mean_net_r": "candidate_daily_mean_net_r",
            "n_obs": "candidate_n_obs",
            "mean_atr_pct": "candidate_mean_atr_pct",
        }
    )

    p = _daily_means(production).rename(
        columns={
            "daily_mean_net_r": "production_daily_mean_net_r",
            "n_obs": "production_n_obs",
            "mean_atr_pct": "production_mean_atr_pct",
        }
    )

    paired = c.merge(
        p,
        on="day_id",
        how="inner",
    )

    if paired.empty:
        paired["difference_daily_mean_net_r"] = []
    else:
        paired["difference_daily_mean_net_r"] = (
            paired["candidate_daily_mean_net_r"]
            - paired["production_daily_mean_net_r"]
        )

    return paired.sort_values(
        "day_id"
    ).reset_index(drop=True)


def _block_bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(values) == 0:
        raise ValueError(
            "cannot bootstrap an empty block sample"
        )

    n_blocks = len(values)

    out = np.empty(
        BOOTSTRAP_ROUNDS,
        dtype=float,
    )

    for i in range(BOOTSTRAP_ROUNDS):
        sampled = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )
        out[i] = float(
            np.mean(
                values[sampled]
            )
        )

    return out


def _paired_bootstrap(
    paired: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """Bootstrap paired daily means using identical sampled day indices."""

    c = paired[
        "candidate_daily_mean_net_r"
    ].to_numpy(dtype=float)

    p = paired[
        "production_daily_mean_net_r"
    ].to_numpy(dtype=float)

    d = paired[
        "difference_daily_mean_net_r"
    ].to_numpy(dtype=float)

    n_blocks = len(paired)

    if n_blocks == 0:
        raise ValueError(
            "cannot run paired bootstrap with zero paired blocks"
        )

    candidate_boot = np.empty(
        BOOTSTRAP_ROUNDS,
        dtype=float,
    )

    production_boot = np.empty(
        BOOTSTRAP_ROUNDS,
        dtype=float,
    )

    difference_boot = np.empty(
        BOOTSTRAP_ROUNDS,
        dtype=float,
    )

    for i in range(BOOTSTRAP_ROUNDS):
        sampled = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        candidate_boot[i] = float(
            np.mean(c[sampled])
        )

        production_boot[i] = float(
            np.mean(p[sampled])
        )

        # Critical: same resampled day indices for both models.
        difference_boot[i] = float(
            np.mean(d[sampled])
        )

    low = (
        (1.0 - CONF) / 2.0
    ) * 100.0

    high = (
        1.0 - (1.0 - CONF) / 2.0
    ) * 100.0

    return {
        "candidate_bootstrap": candidate_boot,
        "production_bootstrap": production_boot,
        "difference_bootstrap": difference_boot,
        "candidate_ci": [
            float(
                np.percentile(
                    candidate_boot,
                    low,
                )
            ),
            float(
                np.percentile(
                    candidate_boot,
                    high,
                )
            ),
        ],
        "production_ci": [
            float(
                np.percentile(
                    production_boot,
                    low,
                )
            ),
            float(
                np.percentile(
                    production_boot,
                    high,
                )
            ),
        ],
        "difference_ci": [
            float(
                np.percentile(
                    difference_boot,
                    low,
                )
            ),
            float(
                np.percentile(
                    difference_boot,
                    high,
                )
            ),
        ],
    }


def _parse_datetime(value):
    if value is None:
        return None

    try:
        ts = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )
    except Exception:
        return None

    if pd.isna(ts):
        return None

    return ts.to_pydatetime()


def _checkpoint_cutoff(
    candidate: pd.DataFrame,
    production: pd.DataFrame,
) -> tuple[str, int]:
    values = []

    for frame in (candidate, production):
        for value in frame.get(
            "resolved_at",
            [],
        ):
            parsed = _parse_datetime(value)

            if parsed is None:
                continue

            values.append(parsed)

    if not values:
        raise ValueError(
            "Checkpoint reached but no valid resolved_at "
            "timestamps exist in the clean observation ledger."
        )

    latest = max(values)

    return (
        latest.isoformat(),
        int(
            latest.timestamp()
            * 1000
        ),
    )


def _load_audit_events(
    path: Path,
    experiment_id: str,
) -> list[dict]:
    if not path.exists():
        return []

    events = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_no, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid experiment ledger JSON at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if (
                record.get(
                    "experiment_id"
                )
                == experiment_id
            ):
                events.append(record)

    return events


def _append_audit_event_once(
    path: Path,
    event: str,
    experiment_id: str,
    payload: dict,
) -> dict:
    events = _load_audit_events(
        path,
        experiment_id,
    )

    matching = [
        item
        for item in events
        if item.get("event") == event
        and item.get("checkpoint_blocks")
        == payload.get("checkpoint_blocks")
    ]

    if matching:
        existing = matching[-1]

        comparable_existing = dict(existing)
        comparable_existing.pop(
            "timestamp",
            None,
        )
        comparable_new = dict(payload)

        if comparable_existing != {
            **{
                "event": event,
                "experiment_id": experiment_id,
            },
            **comparable_new,
        }:
            raise RuntimeError(
                "Existing checkpoint event conflicts with "
                "the newly computed checkpoint payload."
            )

        return existing

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "event": event,
        "experiment_id": experiment_id,
        **payload,
    }

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    return record


def _load_experiment_started_at(
    path: Path,
):
    if not path.exists():
        return None

    try:
        state = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return _parse_datetime(
        state.get(
            "experiment_started_at"
        )
    )


def _load_regime_reference(
    path: Path,
    experiment_id: str,
) -> tuple[dict | None, str]:
    if not path.exists():
        return (
            None,
            "REFERENCE_MISSING",
        )

    try:
        reference = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        return (
            None,
            f"REFERENCE_INVALID_JSON:{exc}",
        )

    if (
        reference.get(
            "experiment_id"
        )
        != experiment_id
    ):
        return (
            None,
            "REFERENCE_EXPERIMENT_ID_MISMATCH",
        )

    cutoffs = reference.get(
        "atr_pct_tercile_cutoffs"
    )

    if (
        not isinstance(
            cutoffs,
            list,
        )
        or len(cutoffs) != 2
    ):
        return (
            None,
            "REFERENCE_CUTOFFS_MISSING",
        )

    try:
        low = float(
            cutoffs[0]
        )
        high = float(
            cutoffs[1]
        )
    except Exception:
        return (
            None,
            "REFERENCE_CUTOFFS_INVALID",
        )

    if not (
        math.isfinite(low)
        and math.isfinite(high)
        and low < high
    ):
        return (
            None,
            "REFERENCE_CUTOFFS_INVALID",
        )

    return (
        {
            **reference,
            "atr_pct_tercile_cutoffs": [
                low,
                high,
            ],
        },
        "READY",
    )


def _assign_volatility_tercile(
    value,
    cutoffs,
):
    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    low, high = cutoffs

    if value < low:
        return "LOW"

    if value < high:
        return "MID"

    return "HIGH"


def _coverage_gate(
    paired: pd.DataFrame,
    candidate: pd.DataFrame,
    production: pd.DataFrame,
    experiment_id: str,
    regime_reference_path: Path,
) -> dict:
    if len(paired) < MIN_PAIRED_BLOCKS:
        return {
            "ready": False,
            "reason": "INSUFFICIENT_PAIRED_BLOCKS",
        }

    final_days = paired.tail(
        MIN_PAIRED_BLOCKS
    ).copy()

    weekend = 0
    weekday = 0

    for day_id in final_days[
        "day_id"
    ]:
        dt = datetime.fromtimestamp(
            (
                int(day_id)
                * DAY_MS
            )
            / 1000.0,
            tz=timezone.utc,
        )

        if dt.weekday() >= 5:
            weekend += 1
        else:
            weekday += 1

    weekend_ok = weekend >= 3
    weekday_ok = weekday >= 10

    reference, reference_status = (
        _load_regime_reference(
            regime_reference_path,
            experiment_id,
        )
    )

    regime_details = {}

    if reference is not None:
        cutoffs = reference[
            "atr_pct_tercile_cutoffs"
        ]

        for model_name, frame in (
            (
                "candidate",
                candidate,
            ),
            (
                "production",
                production,
            ),
        ):
            model_days = _daily_means(
                frame
            )

            model_days = model_days[
                model_days["day_id"].isin(
                    final_days["day_id"]
                )
            ].copy()

            model_days[
                "tercile"
            ] = model_days[
                "mean_atr_pct"
            ].map(
                lambda x: _assign_volatility_tercile(
                    x,
                    cutoffs,
                )
            )

            counts = (
                model_days[
                    "tercile"
                ]
                .value_counts()
                .to_dict()
            )

            represented = sum(
                1
                for label in (
                    "LOW",
                    "MID",
                    "HIGH",
                )
                if counts.get(
                    label,
                    0,
                )
                >= math.ceil(
                    0.15
                    * len(final_days)
                )
            )

            regime_details[
                model_name
            ] = {
                "counts": {
                    label: int(
                        counts.get(
                            label,
                            0,
                        )
                    )
                    for label in (
                        "LOW",
                        "MID",
                        "HIGH",
                    )
                },
                "represented_terciles": represented,
                "required_terciles": 2,
                "ready": (
                    represented >= 2
                ),
            }

    regime_ready = (
        reference is not None
        and all(
            details.get(
                "ready",
                False,
            )
            for details in regime_details.values()
        )
    )

    return {
        "ready": (
            weekend_ok
            and weekday_ok
            and regime_ready
        ),
        "weekend_blocks": weekend,
        "weekday_blocks": weekday,
        "weekend_requirement": 3,
        "weekday_requirement": 10,
        "weekend_ok": weekend_ok,
        "weekday_ok": weekday_ok,
        "regime_reference_status": reference_status,
        "regime_reference_file": str(
            regime_reference_path
        ),
        "regime_details": regime_details,
        "final_15_block_count": len(
            final_days
        ),
        "final_15_day_ids": [
            int(x)
            for x in final_days[
                "day_id"
            ].tolist()
        ],
    }


def _filter_by_cutoff(
    frame: pd.DataFrame,
    cutoff_ms: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    parsed = pd.to_datetime(
        frame["resolved_at"],
        utc=True,
        errors="coerce",
    )

    cutoff = pd.to_datetime(
        cutoff_ms,
        unit="ms",
        utc=True,
    )

    return frame[
        parsed.notna()
        & (parsed <= cutoff)
    ].copy()


def _decision_from_checkpoint(
    candidate: pd.DataFrame,
    production: pd.DataFrame,
    paired: pd.DataFrame,
    checkpoint_blocks: int,
    cutoff_iso: str,
    cutoff_ms: int,
    experiment_id: str,
    audit_ledger_path: Path,
    regime_reference_path: Path,
) -> dict:
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + checkpoint_blocks
    )

    boot = _paired_bootstrap(
        paired,
        rng,
    )

    candidate_ci = boot[
        "candidate_ci"
    ]

    production_ci = boot[
        "production_ci"
    ]

    difference_ci = boot[
        "difference_ci"
    ]

    candidate_profitable = (
        candidate_ci[0] > 0.0
    )

    candidate_beats_production = (
        difference_ci[0] > 0.0
    )

    early_harm = (
        difference_ci[1] < 0.0
    )

    coverage = _coverage_gate(
        paired,
        candidate,
        production,
        experiment_id,
        regime_reference_path,
    )

    if early_harm:
        decision = (
            "EARLY_HARM_STOP"
        )

    elif (
        candidate_profitable
        and candidate_beats_production
        and coverage["ready"]
    ):
        decision = (
            "EFFICACY_PASS"
        )

    elif checkpoint_blocks == FINAL_CHECKPOINT:
        decision = (
            "FINAL_CHECKPOINT_NO_EFFICACY"
        )

    else:
        decision = (
            "CHECKPOINT_CONTINUE"
        )

    payload = {
        "checkpoint_blocks": checkpoint_blocks,
        "cutoff_resolved_at": cutoff_iso,
        "cutoff_resolved_at_ms": cutoff_ms,
        "paired_blocks": int(
            len(paired)
        ),
        "candidate_clean_n": int(
            len(candidate)
        ),
        "production_clean_n": int(
            len(production)
        ),
        "alpha_total": ALPHA_TOTAL,
        "alpha_spent_this_look": ALPHA_PER_CHECKPOINT,
        "confidence_level": CONF,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "bootstrap_seed": (
            BOOTSTRAP_SEED
            + checkpoint_blocks
        ),
        "candidate_ci": [
            float(x)
            for x in candidate_ci
        ],
        "production_ci": [
            float(x)
            for x in production_ci
        ],
        "difference_ci": [
            float(x)
            for x in difference_ci
        ],
        "candidate_profitable": bool(
            candidate_profitable
        ),
        "candidate_beats_production": bool(
            candidate_beats_production
        ),
        "early_harm": bool(
            early_harm
        ),
        "coverage": coverage,
        "decision": decision,
    }

    event = _append_audit_event_once(
        audit_ledger_path,
        CHECKPOINT_EVENT,
        experiment_id,
        payload,
    )

    if decision == "EARLY_HARM_STOP":
        _append_audit_event_once(
            audit_ledger_path,
            HARM_EVENT,
            experiment_id,
            {
                "checkpoint_blocks": checkpoint_blocks,
                "cutoff_resolved_at": cutoff_iso,
                "reason": (
                    "candidate_minus_production "
                    "upper confidence bound < 0"
                ),
                "difference_ci": [
                    float(x)
                    for x in difference_ci
                ],
            },
        )

    return event


def _existing_checkpoint_map(
    events: list[dict],
) -> dict[int, dict]:
    out = {}

    for event in events:
        if (
            event.get("event")
            != CHECKPOINT_EVENT
        ):
            continue

        checkpoint = event.get(
            "checkpoint_blocks"
        )

        try:
            checkpoint = int(
                checkpoint
            )
        except Exception:
            raise RuntimeError(
                "Invalid checkpoint_blocks in audit ledger."
            )

        if checkpoint in out:
            raise RuntimeError(
                "Duplicate checkpoint event detected for "
                f"{checkpoint} blocks."
            )

        out[checkpoint] = event

    return out


def _active_harm_stop(
    events: list[dict],
) -> dict | None:
    stops = [
        event
        for event in events
        if event.get("event")
        == HARM_EVENT
    ]

    if not stops:
        return None

    return stops[-1]


def _result_base(
    candidate: pd.DataFrame,
    production: pd.DataFrame,
    paired: pd.DataFrame,
    candidate_blocks: int,
    production_blocks: int,
    audit_events: list[dict],
    experiment_id: str,
) -> dict:
    return {
        "n_candidate": int(
            len(candidate)
        ),
        "n_production": int(
            len(production)
        ),
        "unique_blocks_candidate": candidate_blocks,
        "unique_blocks_production": production_blocks,
        "paired_unique_blocks": int(
            len(paired)
        ),
        "min_resolved": MIN_RESOLVED,
        "min_unique_blocks": MIN_INDIVIDUAL_BLOCKS,
        "min_paired_blocks": MIN_PAIRED_BLOCKS,
        "checkpoint_blocks": list(
            CHECKPOINT_BLOCKS
        ),
        "final_checkpoint": FINAL_CHECKPOINT,
        "bootstrap_method": (
            "paired_utc_day_bootstrap_of_daily_mean_net_r"
        ),
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONF,
        "alpha_total": ALPHA_TOTAL,
        "alpha_per_checkpoint": ALPHA_PER_CHECKPOINT,
        "block_ms": BLOCK_MS,
        "block_definition": (
            "UTC calendar day keyed by "
            "prediction open_time; "
            "daily statistic is mean Net-R"
        ),
        "paired_day_definition": (
            "candidate and production daily means "
            "on their common UTC calendar days"
        ),
        "experiment_id": experiment_id,
        "checkpoints_evaluated": sorted(
            int(k)
            for k in _existing_checkpoint_map(
                audit_events
            )
        ),
    }


def evaluate_promotion(
    candidate_resolved: list[dict],
    production_resolved: list[dict],
    *,
    experiment_id: str = "PHASE2D-HARDENED-20260924-F08",
    audit_ledger_path: Path = AUDIT_LEDGER_FILE,
    state_path: Path = STATE_FILE,
    regime_reference_path: Path = REGIME_REFERENCE_FILE,
) -> dict:
    """Evaluate the locked prospective gate without continuous peeking."""

    candidate = _clean(
        candidate_resolved
    )

    production = _clean(
        production_resolved
    )

    candidate_blocks = _unique_block_count(
        candidate
    )

    production_blocks = _unique_block_count(
        production
    )

    paired = _paired_daily_means(
        candidate,
        production,
    )

    paired_blocks = int(
        len(paired)
    )

    base = _result_base(
        candidate,
        production,
        paired,
        candidate_blocks,
        production_blocks,
        _load_audit_events(
            audit_ledger_path,
            experiment_id,
        ),
        experiment_id,
    )

    sample_ok = (
        len(candidate) >= MIN_RESOLVED
        and len(production) >= MIN_RESOLVED
    )

    diversity_ok = (
        candidate_blocks >= MIN_INDIVIDUAL_BLOCKS
        and production_blocks >= MIN_INDIVIDUAL_BLOCKS
        and paired_blocks >= MIN_PAIRED_BLOCKS
    )

    events = _load_audit_events(
        audit_ledger_path,
        experiment_id,
    )

    harm_stop = _active_harm_stop(
        events
    )

    if harm_stop is not None:
        return {
            **base,
            "promotion_ready": False,
            "gate_status": "FAIL",
            "sample_gate": sample_ok,
            "diversity_gate": diversity_ok,
            "candidate_sample_ok": (
                len(candidate)
                >= MIN_RESOLVED
            ),
            "production_sample_ok": (
                len(production)
                >= MIN_RESOLVED
            ),
            "candidate_diversity_ok": (
                candidate_blocks
                >= MIN_INDIVIDUAL_BLOCKS
            ),
            "production_diversity_ok": (
                production_blocks
                >= MIN_INDIVIDUAL_BLOCKS
            ),
            "paired_diversity_ok": (
                paired_blocks
                >= MIN_PAIRED_BLOCKS
            ),
            "reason": (
                "F08 early-harm stop already recorded"
            ),
            "experiment_stopped": True,
            "harm_stop": harm_stop,
        }

    if not sample_ok or not diversity_ok:
        reasons = []

        if len(candidate) < MIN_RESOLVED:
            reasons.append(
                f"candidate N={len(candidate)} < {MIN_RESOLVED}"
            )

        if len(production) < MIN_RESOLVED:
            reasons.append(
                f"production N={len(production)} < {MIN_RESOLVED}"
            )

        if candidate_blocks < MIN_INDIVIDUAL_BLOCKS:
            reasons.append(
                "candidate unique_blocks="
                f"{candidate_blocks} < "
                f"{MIN_INDIVIDUAL_BLOCKS}"
            )

        if production_blocks < MIN_INDIVIDUAL_BLOCKS:
            reasons.append(
                "production unique_blocks="
                f"{production_blocks} < "
                f"{MIN_INDIVIDUAL_BLOCKS}"
            )

        if paired_blocks < MIN_PAIRED_BLOCKS:
            reasons.append(
                "paired unique_blocks="
                f"{paired_blocks} < "
                f"{MIN_PAIRED_BLOCKS}"
            )

        return {
            **base,
            "promotion_ready": False,
            "gate_status": "NOT_READY",
            "sample_gate": sample_ok,
            "diversity_gate": diversity_ok,
            "candidate_sample_ok": (
                len(candidate)
                >= MIN_RESOLVED
            ),
            "production_sample_ok": (
                len(production)
                >= MIN_RESOLVED
            ),
            "candidate_diversity_ok": (
                candidate_blocks
                >= MIN_INDIVIDUAL_BLOCKS
            ),
            "production_diversity_ok": (
                production_blocks
                >= MIN_INDIVIDUAL_BLOCKS
            ),
            "paired_diversity_ok": (
                paired_blocks
                >= MIN_PAIRED_BLOCKS
            ),
            "reason": "; ".join(
                reasons
            ),
            "experiment_stopped": False,
            "new_checkpoint_evaluations": [],
        }

    cutoff_iso, cutoff_ms = _checkpoint_cutoff(
        candidate,
        production,
    )

    checkpoint_map = _existing_checkpoint_map(
        events
    )

    reached = [
        checkpoint
        for checkpoint in CHECKPOINT_BLOCKS
        if paired_blocks >= checkpoint
        and checkpoint not in checkpoint_map
    ]

    new_checkpoint_evaluations = []

    for checkpoint in reached:
        c_at = _filter_by_cutoff(
            candidate,
            cutoff_ms,
        )

        p_at = _filter_by_cutoff(
            production,
            cutoff_ms,
        )

        paired_at = _paired_daily_means(
            c_at,
            p_at,
        )

        event = _decision_from_checkpoint(
            c_at,
            p_at,
            paired_at,
            checkpoint,
            cutoff_iso,
            cutoff_ms,
            experiment_id,
            audit_ledger_path,
            regime_reference_path,
        )

        new_checkpoint_evaluations.append(
            event
        )

        # Keep returned checkpoint diagnostics synchronized
        # with checkpoints evaluated in this invocation.
        base["checkpoints_evaluated"] = sorted(
            set(
                base.get(
                    "checkpoints_evaluated",
                    [],
                )
            )
            | {
                int(item["checkpoint_blocks"])
                for item in new_checkpoint_evaluations
            }
        )

        if event.get(
            "decision"
        ) == "EARLY_HARM_STOP":
            return {
                **base,
                "promotion_ready": False,
                "gate_status": "FAIL",
                "sample_gate": True,
                "diversity_gate": True,
                "candidate_sample_ok": True,
                "production_sample_ok": True,
                "candidate_diversity_ok": True,
                "production_diversity_ok": True,
                "paired_diversity_ok": True,
                "reason": (
                    "candidate relative harm at "
                    f"{checkpoint}-block checkpoint"
                ),
                "experiment_stopped": True,
                "new_checkpoint_evaluations": (
                    new_checkpoint_evaluations
                ),
            }

        if event.get(
            "decision"
        ) == "EFFICACY_PASS":
            return {
                **base,
                "promotion_ready": True,
                "gate_status": "PASS",
                "sample_gate": True,
                "diversity_gate": True,
                "candidate_sample_ok": True,
                "production_sample_ok": True,
                "candidate_diversity_ok": True,
                "production_diversity_ok": True,
                "paired_diversity_ok": True,
                "reason": (
                    "paired checkpoint efficacy passed "
                    f"at {checkpoint} blocks"
                ),
                "experiment_stopped": False,
                "decision_checkpoint": checkpoint,
                "new_checkpoint_evaluations": (
                    new_checkpoint_evaluations
                ),
            }

    # No newly evaluated checkpoint. Do not continuously re-peek.
    if new_checkpoint_evaluations:
        latest = new_checkpoint_evaluations[-1]

        decision = latest.get(
            "decision"
        )

        if decision == "FINAL_CHECKPOINT_NO_EFFICACY":
            return {
                **base,
                "promotion_ready": False,
                "gate_status": "FAIL",
                "sample_gate": True,
                "diversity_gate": True,
                "candidate_sample_ok": True,
                "production_sample_ok": True,
                "candidate_diversity_ok": True,
                "production_diversity_ok": True,
                "paired_diversity_ok": True,
                "reason": (
                    "final 30-block checkpoint reached "
                    "without efficacy"
                ),
                "experiment_stopped": True,
                "decision_checkpoint": FINAL_CHECKPOINT,
                "new_checkpoint_evaluations": (
                    new_checkpoint_evaluations
                ),
            }

    started = _load_experiment_started_at(
        state_path
    )

    age_days = None

    if started is not None:
        age_days = (
            datetime.now(
                timezone.utc
            )
            - started
        ).days

        if age_days >= MAX_CALENDAR_DAYS:
            return {
                **base,
                "promotion_ready": False,
                "gate_status": "FAIL",
                "sample_gate": sample_ok,
                "diversity_gate": diversity_ok,
                "candidate_sample_ok": (
                    len(candidate)
                    >= MIN_RESOLVED
                ),
                "production_sample_ok": (
                    len(production)
                    >= MIN_RESOLVED
                ),
                "candidate_diversity_ok": (
                    candidate_blocks
                    >= MIN_INDIVIDUAL_BLOCKS
                ),
                "production_diversity_ok": (
                    production_blocks
                    >= MIN_INDIVIDUAL_BLOCKS
                ),
                "paired_diversity_ok": (
                    paired_blocks
                    >= MIN_PAIRED_BLOCKS
                ),
                "reason": (
                    "90-calendar-day hard ceiling reached "
                    "without a qualifying efficacy decision"
                ),
                "experiment_stopped": True,
                "calendar_age_days": age_days,
                "new_checkpoint_evaluations": (
                    new_checkpoint_evaluations
                ),
            }

    latest_checkpoint = max(
        checkpoint_map.keys(),
        default=None,
    )

    return {
        **base,
        "promotion_ready": False,
        "gate_status": "NOT_READY",
        "sample_gate": True,
        "diversity_gate": True,
        "candidate_sample_ok": True,
        "production_sample_ok": True,
        "candidate_diversity_ok": True,
        "production_diversity_ok": True,
        "paired_diversity_ok": True,
        "reason": (
            "No new fixed checkpoint reached; "
            "continuous peeking is disabled."
        ),
        "experiment_stopped": False,
        "latest_evaluated_checkpoint": latest_checkpoint,
        "new_checkpoint_evaluations": (
            new_checkpoint_evaluations
        ),
        "calendar_age_days": age_days,
    }
