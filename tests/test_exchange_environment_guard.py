from __future__ import annotations

import pytest

from exchange_environment_guard import (
    DERIBIT_TESTNET_BASE,
    EnvironmentGuardError,
    TESTNET_MUTATION_ACK,
    validate_deribit_testnet_mutation_environment,
)


def _valid_env():
    return {
        "DERIBIT_ENVIRONMENT": "TESTNET",
        "DERIBIT_BASE_URL": DERIBIT_TESTNET_BASE,
        "DERIBIT_TESTNET_CLIENT_ID": "testnet-client-id",
        "DERIBIT_TESTNET_CLIENT_SECRET": "testnet-client-secret",
        "DERIBIT_TESTNET_MUTATIONS_ENABLED": "true",
        "DERIBIT_TESTNET_MUTATION_ACK": TESTNET_MUTATION_ACK,
    }


def test_testnet_mutation_environment_accepts_only_explicit_pinned_testnet():
    resolved = validate_deribit_testnet_mutation_environment(environ=_valid_env())
    assert resolved.client_id == "testnet-client-id"
    assert resolved.client_secret == "testnet-client-secret"
    assert resolved.base_url == DERIBIT_TESTNET_BASE


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("DERIBIT_ENVIRONMENT", "PRODUCTION", "must_be_TESTNET"),
        ("DERIBIT_BASE_URL", "https://www.deribit.com/api/v2", "not_pinned_to_testnet"),
        ("DERIBIT_BASE_URL", "http://test.deribit.com/api/v2", "not_pinned_to_testnet"),
        ("DERIBIT_TESTNET_MUTATIONS_ENABLED", "false", "flag_not_enabled"),
        ("DERIBIT_TESTNET_MUTATION_ACK", "yes", "acknowledgement_missing"),
        ("DERIBIT_TESTNET_CLIENT_SECRET", "", "required_environment_missing"),
        ("DERIBIT_CLIENT_SECRET", "legacy-secret", "generic_deribit_credentials_forbidden"),
        ("DERIBIT_MAINNET_CLIENT_ID", "mainnet-id", "production_credentials_must_not_share"),
    ],
)
def test_testnet_mutation_environment_rejects_unsafe_configuration(key, value, message):
    env = _valid_env()
    env[key] = value
    with pytest.raises(EnvironmentGuardError, match=message):
        validate_deribit_testnet_mutation_environment(environ=env)


def test_testnet_mutation_environment_requires_constructor_credentials_to_match_env():
    with pytest.raises(EnvironmentGuardError, match="client_secret_does_not_match"):
        validate_deribit_testnet_mutation_environment(
            environ=_valid_env(),
            expected_client_id="testnet-client-id",
            expected_client_secret="wrong-secret",
        )


def test_testnet_mutation_environment_rejects_production_host_with_testnet_label():
    env = _valid_env()
    env["DERIBIT_BASE_URL"] = "https://www.deribit.com/api/v2"
    with pytest.raises(EnvironmentGuardError):
        validate_deribit_testnet_mutation_environment(environ=env)
