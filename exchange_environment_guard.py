"""Fail-closed environment and credential guard for Deribit Testnet mutations.

This module is intentionally independent of DeribitClient so the constructor,
scanner and focused tests can share one validation contract without a cycle.
It does not enable mutations; callers must separately pass their code-level and
workflow-level approval gates.
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlsplit


DERIBIT_TESTNET_BASE = "https://test.deribit.com/api/v2"
TESTNET_MUTATION_ACK = "I_ACKNOWLEDGE_DERIBIT_TESTNET_ONLY"


class EnvironmentGuardError(RuntimeError):
    """Environment/credential mismatch; safe to log without secret values."""


@dataclass(frozen=True)
class DeribitTestnetMutationCredentials:
    client_id: str
    client_secret: str
    base_url: str = DERIBIT_TESTNET_BASE


def _required(env: Mapping[str, str], key: str) -> str:
    value = str(env.get(key, "") or "").strip()
    if not value:
        raise EnvironmentGuardError(f"required_environment_missing:{key}")
    return value


def validate_deribit_testnet_mutation_environment(
    *,
    environ: Optional[Mapping[str, str]] = None,
    expected_client_id: Optional[str] = None,
    expected_client_secret: Optional[str] = None,
) -> DeribitTestnetMutationCredentials:
    """Validate that a write-scoped client can only target a confirmed Testnet.

    Required independent controls:
      * DERIBIT_ENVIRONMENT=TESTNET
      * DERIBIT_BASE_URL exactly pinned to test.deribit.com
      * dedicated DERIBIT_TESTNET_CLIENT_ID/SECRET variables
      * explicit DERIBIT_TESTNET_MUTATIONS_ENABLED=true
      * exact human acknowledgement value
      * no legacy generic or explicitly named production credentials in scope

    If credentials are passed by a caller, they must match the dedicated
    environment values. The function never returns or logs secret values except
    in its return object for immediate client construction.
    """
    env = os.environ if environ is None else environ

    environment = _required(env, "DERIBIT_ENVIRONMENT").upper()
    if environment != "TESTNET":
        raise EnvironmentGuardError("deribit_environment_must_be_TESTNET")

    base_url = _required(env, "DERIBIT_BASE_URL")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "test.deribit.com"
        or parsed.path.rstrip("/") != "/api/v2"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EnvironmentGuardError("deribit_base_url_not_pinned_to_testnet")
    if base_url.rstrip("/") != DERIBIT_TESTNET_BASE:
        raise EnvironmentGuardError("deribit_base_url_not_exact_testnet_endpoint")

    if str(env.get("DERIBIT_TESTNET_MUTATIONS_ENABLED", "false")).strip().lower() != "true":
        raise EnvironmentGuardError("deribit_testnet_mutation_flag_not_enabled")
    if not hmac.compare_digest(
        str(env.get("DERIBIT_TESTNET_MUTATION_ACK", "")),
        TESTNET_MUTATION_ACK,
    ):
        raise EnvironmentGuardError("deribit_testnet_mutation_acknowledgement_missing")

    generic_keys = ("DERIBIT_CLIENT_ID", "DERIBIT_CLIENT_SECRET")
    if any(str(env.get(key, "") or "").strip() for key in generic_keys):
        raise EnvironmentGuardError("legacy_generic_deribit_credentials_forbidden")

    production_keys = (
        "DERIBIT_MAINNET_CLIENT_ID",
        "DERIBIT_MAINNET_CLIENT_SECRET",
        "DERIBIT_PRODUCTION_CLIENT_ID",
        "DERIBIT_PRODUCTION_CLIENT_SECRET",
    )
    if any(str(env.get(key, "") or "").strip() for key in production_keys):
        raise EnvironmentGuardError("production_credentials_must_not_share_testnet_process")

    client_id = _required(env, "DERIBIT_TESTNET_CLIENT_ID")
    client_secret = _required(env, "DERIBIT_TESTNET_CLIENT_SECRET")

    if expected_client_id is not None and not hmac.compare_digest(
        str(expected_client_id), client_id
    ):
        raise EnvironmentGuardError("client_id_does_not_match_testnet_environment")
    if expected_client_secret is not None and not hmac.compare_digest(
        str(expected_client_secret), client_secret
    ):
        raise EnvironmentGuardError("client_secret_does_not_match_testnet_environment")

    return DeribitTestnetMutationCredentials(client_id, client_secret)
