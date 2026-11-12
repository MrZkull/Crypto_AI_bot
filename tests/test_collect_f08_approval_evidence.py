from __future__ import annotations

import json
from io import BytesIO

import pytest

from tools import collect_f08_approval_evidence as collector


def _api_environment(environment: str):
    return {
        "name": environment,
        "protection_rules": [
            {
                "type": "required_reviewers",
                "prevent_self_review": True,
                "reviewers": [
                    {"type": "User", "reviewer": {"login": "MrZkull", "id": 166250462}}
                ],
            }
        ],
        "deployment_branch_policy": {
            "protected_branches": True,
            "custom_branch_policies": False,
        },
    }


def _approval_history(environment: str):
    return [
        {
            "state": "approved",
            "user": {"login": "MrZkull", "id": 166250462},
            "environments": [{"name": environment}],
        }
    ]


def _install_context(monkeypatch, environment: str, workflow: str, workflow_file: str):
    values = {
        "GITHUB_REPOSITORY": "MrZkull/Crypto_AI_bot",
        "GITHUB_WORKFLOW": workflow,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_ACTOR": "automation-operator",
        "GITHUB_ACTOR_ID": "987654321",
        "GITHUB_TOKEN": "test-token-do-not-persist",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


@pytest.mark.parametrize(
    "environment,workflow,workflow_file",
    [
        (
            "f08-validator-amendment",
            "Phase2D F08 Validator Amendment",
            ".github/workflows/phase2d_f08_validator_amendment.yml",
        ),
        (
            "f08-disposition",
            "Phase2D F08 Controlled Disposition",
            ".github/workflows/phase2d_f08_controlled_disposition.yml",
        ),
    ],
)
def test_collect_fetches_environment_and_actual_run_approval(monkeypatch, environment, workflow, workflow_file):
    _install_context(monkeypatch, environment, workflow, workflow_file)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if f"/environments/{environment}" in request.full_url:
            payload = _api_environment(environment)
        else:
            payload = _approval_history(environment)
        return BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(collector.urllib.request, "urlopen", fake_urlopen)
    evidence = collector.collect(environment, workflow, workflow_file)

    assert len(requests) == 2
    assert requests[0].get_header("Authorization") == "Bearer test-token-do-not-persist"
    assert requests[1].get_header("Authorization") == "Bearer test-token-do-not-persist"
    assert evidence["context"]["run_id"] == "123456789"
    assert evidence["context"]["actor_id"] == "987654321"
    assert evidence["environment"]["deployment_branch_policy"]["protected_branches"] is True
    assert evidence["approval_history"][0]["user"]["id"] == 166250462
    serialized = json.dumps(evidence)
    assert "test-token-do-not-persist" not in serialized


def test_collector_rejects_non_main_workflow_dispatch(monkeypatch):
    environment = "f08-validator-amendment"
    workflow = "Phase2D F08 Validator Amendment"
    workflow_file = ".github/workflows/phase2d_f08_validator_amendment.yml"
    _install_context(monkeypatch, environment, workflow, workflow_file)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature/unsafe")
    with pytest.raises(collector.ApprovalEvidenceError, match="refs/heads/main"):
        collector.collect(environment, workflow, workflow_file)


def test_collector_rejects_wrong_repository(monkeypatch):
    environment = "f08-validator-amendment"
    workflow = "Phase2D F08 Validator Amendment"
    workflow_file = ".github/workflows/phase2d_f08_validator_amendment.yml"
    _install_context(monkeypatch, environment, workflow, workflow_file)
    monkeypatch.setenv("GITHUB_REPOSITORY", "fork/Crypto_AI_bot")
    with pytest.raises(collector.ApprovalEvidenceError, match="Unexpected repository"):
        collector.collect(environment, workflow, workflow_file)


def test_collector_fails_when_actions_token_cannot_read_environment(monkeypatch):
    environment = "f08-validator-amendment"
    workflow = "Phase2D F08 Validator Amendment"
    workflow_file = ".github/workflows/phase2d_f08_validator_amendment.yml"
    _install_context(monkeypatch, environment, workflow, workflow_file)

    def denied(request, timeout):
        raise OSError("forbidden")

    monkeypatch.setattr(collector.urllib.request, "urlopen", denied)
    with pytest.raises(collector.ApprovalEvidenceError, match="Could not read GitHub environment API"):
        collector.collect(environment, workflow, workflow_file)
