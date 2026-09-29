"""Evidence-only Stage 2 signal arbitration.

Stage 2 remains evidence-only: this module performs no exchange I/O and does
not activate production substitution.

Comparability requires exact:
    logical_symbol + open_time + interval + observation_source

Missing or mismatched provenance fails closed to production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Signal = Literal["BUY", "SELL", "NO_TRADE"]
ObservationSource = tuple[str, str]


@dataclass(frozen=True)
class ObservationKey:
    logical_symbol: str
    open_time: int
    interval: str
    observation_exchange: str
    observation_market_type: str


@dataclass(frozen=True)
class ArbitrationResult:
    final_signal: Signal
    selected_policy: Literal["candidate", "production"]
    reason: str
    comparable: bool


def _normalize_signal(value: str) -> Signal:
    signal = str(value).upper().strip()
    if signal not in {"BUY", "SELL", "NO_TRADE"}:
        raise ValueError(f"Unsupported signal: {value!r}")
    return signal  # type: ignore[return-value]


def _extract_signal(observation: dict, *, candidate: bool) -> str:
    fields = ("signal", "predicted_signal", "side") if candidate else (
        "signal",
        "predicted_signal",
    )
    for field in fields:
        value = observation.get(field)
        if value not in (None, ""):
            return str(value)
    raise KeyError(
        "Observation is missing a normalized signal field; "
        f"checked {fields!r}"
    )


def _normalize_observation_source(value: object) -> ObservationSource | None:
    if not isinstance(value, dict):
        return None

    exchange = str(value.get("exchange", "")).strip().lower()
    market_type = str(value.get("market_type", "")).strip().lower()

    if not exchange or not market_type:
        return None

    return exchange, market_type


def _observation_key(observation: dict) -> ObservationKey | None:
    logical_symbol = observation.get("logical_symbol") or observation.get("symbol")
    interval = observation.get("interval")
    open_time = observation.get("open_time")
    source = _normalize_observation_source(observation.get("observation_source"))

    if logical_symbol in (None, "") or interval in (None, "") or open_time is None:
        return None
    if source is None:
        return None

    try:
        open_time_int = int(open_time)
    except (TypeError, ValueError, OverflowError):
        return None

    interval_norm = str(interval).strip().lower()
    if not interval_norm:
        return None

    exchange, market_type = source
    return ObservationKey(
        logical_symbol=str(logical_symbol).upper().strip(),
        open_time=open_time_int,
        interval=interval_norm,
        observation_exchange=exchange,
        observation_market_type=market_type,
    )


def arbitrate_signals(
    *,
    candidate_symbol: str,
    candidate_open_time: int,
    candidate_signal: str,
    production_symbol: str,
    production_open_time: int,
    production_signal: str,
) -> ArbitrationResult:
    """Apply the evidence-only Stage 2 substitution policy."""
    cand = _normalize_signal(candidate_signal)
    prod = _normalize_signal(production_signal)

    keys_match = (
        str(candidate_symbol).upper() == str(production_symbol).upper()
        and int(candidate_open_time) == int(production_open_time)
    )

    if not keys_match:
        symbol_match = (
            str(candidate_symbol).upper() == str(production_symbol).upper()
        )
        reason = "TIMESTAMP_MISMATCH" if symbol_match else "SYMBOL_MISMATCH"
        return ArbitrationResult(
            final_signal=prod,
            selected_policy="production",
            reason=reason,
            comparable=False,
        )

    if prod == "SELL":
        return ArbitrationResult(
            final_signal="SELL",
            selected_policy="production",
            reason="PRODUCTION_SELL_PRECEDENCE",
            comparable=True,
        )

    if cand == "BUY" and prod == "NO_TRADE":
        return ArbitrationResult(
            final_signal="BUY",
            selected_policy="candidate",
            reason="CANDIDATE_BUY_SUBSTITUTION",
            comparable=True,
        )

    if cand == "BUY" and prod == "BUY":
        return ArbitrationResult(
            final_signal="BUY",
            selected_policy="production",
            reason="BOTH_BUY_NO_POLICY_CHANGE",
            comparable=True,
        )

    return ArbitrationResult(
        final_signal=prod,
        selected_policy="production",
        reason="CANDIDATE_NOT_ELIGIBLE_FOR_SUBSTITUTION",
        comparable=True,
    )


def arbitrate_observations(candidate: dict, production: dict) -> ArbitrationResult:
    """Apply Stage 2 to real candidate/production observation dictionaries.

    Complete provenance agreement is required before candidate substitution.
    """
    prod = _normalize_signal(_extract_signal(production, candidate=False))

    candidate_key = _observation_key(candidate)
    production_key = _observation_key(production)

    if candidate_key is None or production_key is None:
        return ArbitrationResult(
            final_signal=prod,
            selected_policy="production",
            reason="MISSING_OBSERVATION_PROVENANCE",
            comparable=False,
        )

    if candidate_key.logical_symbol != production_key.logical_symbol:
        return ArbitrationResult(
            final_signal=prod,
            selected_policy="production",
            reason="SYMBOL_MISMATCH",
            comparable=False,
        )

    if candidate_key.open_time != production_key.open_time:
        return ArbitrationResult(
            final_signal=prod,
            selected_policy="production",
            reason="TIMESTAMP_MISMATCH",
            comparable=False,
        )

    if candidate_key.interval != production_key.interval:
        return ArbitrationResult(
            final_signal=prod,
            selected_policy="production",
            reason="INTERVAL_MISMATCH",
            comparable=False,
        )

    if (
        candidate_key.observation_exchange != production_key.observation_exchange
        or candidate_key.observation_market_type != production_key.observation_market_type
    ):
        return ArbitrationResult(
            final_signal=prod,
            selected_policy="production",
            reason="OBSERVATION_SOURCE_MISMATCH",
            comparable=False,
        )

    return arbitrate_signals(
        candidate_symbol=candidate_key.logical_symbol,
        candidate_open_time=candidate_key.open_time,
        candidate_signal=_extract_signal(candidate, candidate=True),
        production_symbol=production_key.logical_symbol,
        production_open_time=production_key.open_time,
        production_signal=_extract_signal(production, candidate=False),
    )
