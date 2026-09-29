"""Evidence-only Stage 2 signal arbitration.

This module deliberately contains no exchange I/O and does not modify the
production execution path. It encodes the proposed Stage 2 substitution rule:

* Candidate substitutions are BUY-only.
* A production SELL always retains precedence over candidate BUY.
* Candidate SELL never replaces the production policy.
* Candidate and production observations are comparable only when both symbol
  and open_time match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Signal = Literal["BUY", "SELL", "NO_TRADE"]


@dataclass(frozen=True)
class ObservationKey:
    symbol: str
    open_time: int


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


def arbitrate_signals(
    *,
    candidate_symbol: str,
    candidate_open_time: int,
    candidate_signal: str,
    production_symbol: str,
    production_open_time: int,
    production_signal: str,
) -> ArbitrationResult:
    """Apply the evidence-only Stage 2 substitution policy.

    Exact symbol/open_time agreement is required before candidate substitution
    can occur. Production SELL has absolute precedence. Only candidate BUY can
    substitute for the production policy.
    """
    cand = _normalize_signal(candidate_signal)
    prod = _normalize_signal(production_signal)

    keys_match = (
        str(candidate_symbol).upper() == str(production_symbol).upper()
        and int(candidate_open_time) == int(production_open_time)
    )

    if not keys_match:
        symbol_match = str(candidate_symbol).upper() == str(production_symbol).upper()
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
