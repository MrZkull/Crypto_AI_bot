#!/usr/bin/env python3
"""Collect read-only GitHub protected-environment approval evidence for F08.

Both the environment configuration and workflow-run approval history are
fetched with the run's GITHUB_TOKEN, which needs repository Actions: read.
No separate personal access token is needed. The token is never written to
the evidence file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://api.github.com"
REPOSITORY = "MrZkull/Crypto_AI_bot"


class ApprovalEvidenceError(RuntimeError):
    pass


def _get_json(url: str, token: str, label: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "Crypto-AI-F08-approval-evidence-collector",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise ApprovalEvidenceError(
            f"GitHub {label} API returned HTTP {exc.code}: {detail}"
        ) from exc
    except Exception as exc:
        raise ApprovalEvidenceError(f"Could not read GitHub {label} API: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ApprovalEvidenceError(f"GitHub {label} API returned invalid JSON") from exc


def collect(environment_name: str, workflow_name: str, workflow_file: str) -> dict:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if repository != REPOSITORY:
        raise ApprovalEvidenceError(f"Unexpected repository context: {repository!r}")

    required_env = (
        "GITHUB_WORKFLOW", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_ACTOR", "GITHUB_ACTOR_ID",
        "GITHUB_TOKEN",
    )
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise ApprovalEvidenceError(f"Missing required workflow environment values: {missing}")

    if os.environ["GITHUB_WORKFLOW"] != workflow_name:
        raise ApprovalEvidenceError("Current workflow name does not match the pinned approval purpose")
    if os.environ["GITHUB_EVENT_NAME"] != "workflow_dispatch" or os.environ["GITHUB_REF"] != "refs/heads/main":
        raise ApprovalEvidenceError("Approval evidence can only be collected for workflow_dispatch on refs/heads/main")
    if not os.environ["GITHUB_RUN_ID"].isdigit():
        raise ApprovalEvidenceError("GITHUB_RUN_ID is malformed")
    if not os.environ["GITHUB_RUN_ATTEMPT"].isdigit() or int(os.environ["GITHUB_RUN_ATTEMPT"]) < 1:
        raise ApprovalEvidenceError("GITHUB_RUN_ATTEMPT is malformed")
    if len(os.environ["GITHUB_SHA"]) != 40 or any(ch not in "0123456789abcdef" for ch in os.environ["GITHUB_SHA"]):
        raise ApprovalEvidenceError("GITHUB_SHA is malformed")
    if not os.environ["GITHUB_ACTOR_ID"].isdigit():
        raise ApprovalEvidenceError("GITHUB_ACTOR_ID is malformed")

    run_id = os.environ["GITHUB_RUN_ID"]
    environment = _get_json(
        f"{API_BASE}/repos/{REPOSITORY}/environments/{environment_name}",
        os.environ["GITHUB_TOKEN"],
        "environment",
    )
    approvals = _get_json(
        f"{API_BASE}/repos/{REPOSITORY}/actions/runs/{run_id}/approvals",
        os.environ["GITHUB_TOKEN"],
        "workflow-run approvals",
    )
    if not isinstance(environment, dict) or not isinstance(approvals, list):
        raise ApprovalEvidenceError("GitHub environment or approval-history response has an unexpected shape")

    return {
        "context": {
            "repository": repository,
            "workflow": os.environ["GITHUB_WORKFLOW"],
            "workflow_file": workflow_file,
            "event_name": os.environ["GITHUB_EVENT_NAME"],
            "ref": os.environ["GITHUB_REF"],
            "run_id": run_id,
            "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
            "head_sha": os.environ["GITHUB_SHA"],
            "actor_login": os.environ["GITHUB_ACTOR"],
            "actor_id": os.environ["GITHUB_ACTOR_ID"],
        },
        "environment": environment,
        "approval_history": approvals,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        evidence = collect(args.environment, args.workflow, args.workflow_file)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, args.output)
        print(f"F08 approval evidence collected: {args.output}")
        print("Environment configuration fetched: YES")
        print("Run approval history fetched: YES")
        print("Secrets written to evidence: NO")
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
