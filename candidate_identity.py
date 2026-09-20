#!/usr/bin/env python3
"""Immutable candidate identity/provenance contract.

The required fields match the candidate manifest emitted by train_model.py.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

IDENTITY_FIELDS = (
    "artifact_version",
    "candidate_id",
    "model_sha256",
    "feature_schema_hash",
    "feature_code_hash",
    "execution_policy_hash",
    "config_hash",
    "candidate_created_at",
    "candidate_expiry_at",
    "status",
)
TRADE_IDENTITY_FIELDS = (
    "candidate_id",
    "model_sha256",
    "feature_schema_hash",
    "feature_code_hash",
    "execution_policy_hash",
    "config_hash",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def validate_manifest(m: dict) -> tuple[bool, list[str]]:
    errors = []
    for key in IDENTITY_FIELDS:
        if key not in m or m[key] in (None, ""):
            errors.append(f"missing:{key}")
    return (not errors, errors)


def trade_identity_ok(trade: dict, manifest: dict) -> tuple[bool, list[str]]:
    errors = []
    for key in TRADE_IDENTITY_FIELDS:
        if trade.get(key) != manifest.get(key):
            errors.append(f"mismatch:{key}")
    return (not errors, errors)
