from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phase2d_f08_historical_backfill as backfill


def _make_inputs(root: Path, monkeypatch):
    repo_root = root / "repo"
    research = repo_root / "research_outputs"
    research.mkdir(parents=True)
    input_dir = repo_root / "_test_inputs"
    input_dir.mkdir()
    source_paths = {}
    suffixes = {
        "state": ".json", "ledger": ".jsonl", "journal": ".jsonl",
        "results": ".json", "snapshot": ".json", "proposal": ".json",
        "raw_windows": ".jsonl", "dry_run": ".json", "independent": ".json",
        "actual_resolver": ".json",
    }
    for index, name in enumerate(backfill.EXPECTED_SOURCE_NAMES):
        path = input_dir / f"{name}{suffixes[name]}"
        path.write_bytes(f"source-fixture:{name}:{index}\n".encode())
        source_paths[name] = path
    state = {"pending": {"candidate": [], "production": []}}
    results = []
    status_counts = {
        "candidate": ["TP"] * 12 + ["SL"] * 62,
        "production": ["TP"] * 280 + ["SL"] * 245 + ["EXPIRED"] * 19,
    }
    gross = {"TP": 3.5, "SL": -2.5, "EXPIRED": 0.0}
    net = {"TP": 3.38, "SL": -2.62, "EXPIRED": -0.12}
    for model, count in (("candidate", 74), ("production", 544)):
        for index in range(count):
            prediction_id = f"{model}-{index:04d}"
            if model == "candidate":
                unique_no = index
            elif index < 45:
                unique_no = index
            else:
                unique_no = 74 + (index - 45)
            row = {
                "id": prediction_id,
                "model": model,
                "symbol": f"SYM-{unique_no:04d}",
                "open_time": 1_700_000_000_000 + unique_no * 900_000,
                "created_at": "2026-10-04T12:00:00+00:00",
                "signal": "BUY" if index % 2 == 0 else "SELL",
                "entry": 100.0,
                "atr": 1.0,
            }
            state["pending"][model].append(row)
            status = status_counts[model][index]
            results.append({
                **row,
                "status": status,
                "outcome_bar": 24 if status == "EXPIRED" else 1,
                "gross_r": gross[status],
                "net_r": net[status],
                "resolved_at": "2026-11-05T12:00:00+00:00",
            })

    raw_windows = {}
    for row in state["pending"]["candidate"] + state["pending"]["production"]:
        key = (row["symbol"], int(row["open_time"]))
        raw_windows.setdefault(key, {
            "symbol": row["symbol"],
            "open_time": int(row["open_time"]),
            "request": {"resolution": "15"},
            "candles": [
                {"timestamp": int(row["open_time"]) + j * 900_000, "high": 101.0, "low": 99.0}
                for j in range(26)
            ],
        })
    assert len(raw_windows) == 573
    raw_windows_path = source_paths["raw_windows"]
    raw_windows_path.write_bytes(b"".join(
        backfill.canonical_json_bytes(window) + b"\n"
        for _, window in sorted(raw_windows.items())
    ))
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    monkeypatch.setattr(backfill, "EXPECTED_STATE_SHA256", source_hashes["state"])
    monkeypatch.setattr(backfill, "EXPECTED_LEDGER_SHA256", source_hashes["ledger"])

    resolver_identity = {
        "git_head": backfill.EXPECTED_REVIEW_HEAD,
        "validator_git_blob": backfill.EXPECTED_VALIDATOR_GIT_BLOB,
        "dependency_git_blobs": dict(backfill.EXPECTED_DEPENDENCY_BLOBS),
        "pr14_merge_ancestor": True,
        "pr14_merge_commit": backfill.EXPECTED_PR14_MERGE,
    }
    resolver_verification = {
        "verification": "PASS",
        "verification_type": "actual_production_resolve_prediction",
        "predictions_verified": 618,
        "candidate_verified": 74,
        "production_verified": 544,
        "mismatch_count": 0,
        "state_sha256_before": source_hashes["state"],
        "state_sha256_after": source_hashes["state"],
        "ledger_sha256_before": source_hashes["ledger"],
        "ledger_sha256_after": source_hashes["ledger"],
        "actual_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
    }
    plan_info = {
        "predictions_resolved": 618,
        "candidate": {"TP": 12, "SL": 62},
        "production": {"TP": 280, "SL": 245, "EXPIRED": 19},
        "total_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
        "raw_windows_used": 573,
        "resolution_timestamp_policy": "resolved_at is preserved in the separately hashed per-prediction resolution provenance sidecar",
    }
    journal_info = {"rows": 98, "c15_rows": 1, "c20_rows": 1, "c25_or_later_rows": 0, "f08_amendment_rows": 1}
    before = {name: path.read_bytes() for name, path in source_paths.items()}
    return repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, before


def _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification):
    return backfill.write_historical_backfill_bundle(
        repo_root=repo_root,
        output_dir=repo_root / "research_outputs" / "f08_historical_backfill",
        source_paths=source_paths,
        source_hashes=source_hashes,
        state=state,
        result_records=results,
        plan_info=plan_info,
        journal_info=journal_info,
        resolver_identity=resolver_identity,
        resolver_verification=resolver_verification,
    )


def test_bundle_contains_exactly_618_historical_rows_and_no_state_or_ledger(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, before = values
    outcome = _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification)
    output = repo_root / "research_outputs" / "f08_historical_backfill"
    assert outcome["status"] == "CREATED"
    assert (output / "manifest.json").is_file()
    assert (output / "observations.jsonl").is_file()
    assert (output / backfill.PROVENANCE_FILENAME).is_file()
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    provenance = [json.loads(line) for line in (output / backfill.PROVENANCE_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert len(provenance) == 618
    assert {row["resolved_at"] for row in provenance} == {"2026-11-05T12:00:00+00:00"}
    assert all(row["outcome_bar_close_time_ms"] - row["outcome_bar_open_time_ms"] == 900_000 for row in provenance)
    assert all(row["outcome_bar_close_time"].endswith("Z") for row in provenance)
    assert len(rows) == 618
    assert sum(row["model"] == "candidate" for row in rows) == 74
    assert sum(row["model"] == "production" for row in rows) == 544
    assert {row["evidence_class"] for row in rows} == {"HISTORICAL_BACKFILL"}
    assert {row["reconciliation_id"] for row in rows} == {outcome["reconciliation_id"]}
    assert all("resolved_at" not in row for row in rows)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_status"] == "READY_FOR_OWNER_REVIEW"
    assert manifest["approval_status"] == "PENDING_OWNER_APPROVAL"
    assert manifest["historical_bundle"]["bundle_sha256"] == hashlib.sha256((output / "observations.jsonl").read_bytes()).hexdigest()
    assert manifest["resolution_provenance"]["sha256"] == hashlib.sha256((output / backfill.PROVENANCE_FILENAME).read_bytes()).hexdigest()
    assert manifest["resolution_provenance"]["record_count"] == 618
    assert not (output / "phase2d_state.json").exists()
    assert not (output / "phase2d_observations.jsonl").exists()
    assert manifest["safety_invariants"]["official_state_written"] is False
    assert manifest["safety_invariants"]["official_ledger_written"] is False
    assert manifest["safety_invariants"]["official_event_appended"] is False
    for name, path in source_paths.items():
        assert path.read_bytes() == before[name]


def test_repeated_runs_are_identical_only_when_resolution_provenance_is_identical(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    first = _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification)
    output = repo_root / "research_outputs" / "f08_historical_backfill"
    first_tree = backfill._tree_hashes(output)

    # A changed resolver wall clock must be preserved as changed evidence; the
    # immutable output directory must reject that different sidecar rather than
    # pretending the two provenance records are byte-identical.
    results_again = [dict(row, resolved_at="2026-11-06T12:15:00+00:00") for row in results]
    with pytest.raises(backfill.BackfillBundleError, match="different contents"):
        backfill.write_historical_backfill_bundle(
            repo_root=repo_root,
            output_dir=output,
            source_paths=source_paths,
            source_hashes=source_hashes,
            state=state,
            result_records=results_again,
            plan_info=plan_info,
            journal_info=journal_info,
            resolver_identity=resolver_identity,
            resolver_verification=resolver_verification,
        )
    second = _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification)
    assert second["status"] == "ALREADY_PRESENT_IDENTICAL"
    assert first["reconciliation_id"] == second["reconciliation_id"]
    assert first_tree == backfill._tree_hashes(output)


def test_different_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification)
    output = repo_root / "research_outputs" / "f08_historical_backfill"
    marker = output / "operator-note.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(backfill.BackfillBundleError, match="different contents"):
        _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification)
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_bundle_writer_rejects_noncanonical_output_path(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    with pytest.raises(backfill.BackfillBundleError, match="exactly"):
        backfill.write_historical_backfill_bundle(
            repo_root=repo_root,
            output_dir=repo_root / "research_outputs" / "other",
            source_paths=source_paths,
            source_hashes=source_hashes,
            state=state,
            result_records=results,
            plan_info=plan_info,
            journal_info=journal_info,
            resolver_identity=resolver_identity,
            resolver_verification=resolver_verification,
        )


def test_bundle_writer_rejects_source_symlink_before_resolving(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    alias = tmp_path / "state-alias.json"
    try:
        alias.symlink_to(source_paths["state"])
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment")
    linked_sources = dict(source_paths)
    linked_sources["state"] = alias
    with pytest.raises(backfill.BackfillBundleError, match="Symlink path component"):
        backfill.write_historical_backfill_bundle(
            repo_root=repo_root,
            output_dir=repo_root / "research_outputs" / "f08_historical_backfill",
            source_paths=linked_sources,
            source_hashes=source_hashes,
            state=state,
            result_records=results,
            plan_info=plan_info,
            journal_info=journal_info,
            resolver_identity=resolver_identity,
            resolver_verification=resolver_verification,
        )


def test_bundle_writer_rejects_output_symlink_before_resolving(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    output = repo_root / "research_outputs" / "f08_historical_backfill"
    target = repo_root / "research_outputs" / "alternate-output-target"
    target.mkdir()
    try:
        output.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment")
    with pytest.raises(backfill.BackfillBundleError, match="Symlink path component"):
        backfill.write_historical_backfill_bundle(
            repo_root=repo_root,
            output_dir=output,
            source_paths=source_paths,
            source_hashes=source_hashes,
            state=state,
            result_records=results,
            plan_info=plan_info,
            journal_info=journal_info,
            resolver_identity=resolver_identity,
            resolver_verification=resolver_verification,
        )
    assert list(target.iterdir()) == []


def test_bundle_writer_rejects_repository_root_symlink(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    alias = tmp_path / "repo-alias"
    try:
        alias.symlink_to(repo_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment")
    with pytest.raises(backfill.BackfillBundleError, match="Symlink path component"):
        backfill.write_historical_backfill_bundle(
            repo_root=alias,
            output_dir=alias / "research_outputs" / "f08_historical_backfill",
            source_paths=source_paths,
            source_hashes=source_hashes,
            state=state,
            result_records=results,
            plan_info=plan_info,
            journal_info=journal_info,
            resolver_identity=resolver_identity,
            resolver_verification=resolver_verification,
        )


def test_resolver_loader_rejects_repository_symlink_before_git_access(tmp_path):
    target = tmp_path / "resolver"
    target.mkdir()
    alias = tmp_path / "resolver-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment")
    with pytest.raises(backfill.BackfillBundleError, match="Symlink path component"):
        backfill.load_resolver_module(alias)


def test_resolution_plan_rejects_resolver_identity_mutation():
    import pandas as pd
    state, proposal, windows, resolver = make_fixture()

    def mutating_resolver(pred, frame):
        result = resolver(pred, frame)
        result["symbol"] = "OTHER_SYMBOL"
        return result

    with pytest.raises(backfill.BackfillBundleError, match="changed source prediction identity/payload"):
        backfill.build_resolution_plan(state, proposal, windows, mutating_resolver, pd)


def test_resolution_plan_rejects_non_utc_resolution_timestamp():
    import pandas as pd
    state, proposal, windows, resolver = make_fixture()

    def non_utc_resolver(pred, frame):
        result = resolver(pred, frame)
        result["resolved_at"] = "2026-11-04T17:30:00+05:30"
        return result

    with pytest.raises(backfill.BackfillBundleError, match="explicit UTC offset"):
        backfill.build_resolution_plan(state, proposal, windows, non_utc_resolver, pd)


def test_resolution_plan_rejects_pre_cutoff_resolution_timestamp():
    import pandas as pd
    state, proposal, windows, resolver = make_fixture()

    def old_timestamp_resolver(pred, frame):
        result = resolver(pred, frame)
        result["resolved_at"] = "2026-10-16T14:08:36.946361+00:00"
        return result

    with pytest.raises(backfill.BackfillBundleError, match="after the legacy cutoff"):
        backfill.build_resolution_plan(state, proposal, windows, old_timestamp_resolver, pd)


def test_bundle_writer_rejects_nonstale_or_wrong_count_pending(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path, monkeypatch)
    repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification, _ = values
    state["pending"]["candidate"][0]["created_at"] = "2026-10-17T00:00:00+00:00"
    with pytest.raises(backfill.BackfillBundleError, match="not stale"):
        _write(repo_root, source_paths, source_hashes, state, results, plan_info, journal_info, resolver_identity, resolver_verification)


# Reuse the reviewed 618-prediction fixture shape to test the real resolver-plan contract.


def make_fixture():
    """Build exactly 618 predictions across 573 unique candle windows."""
    state = {
        "schema_version": 6,
        "experiment_definition": {"experiment_id": backfill.EXPERIMENT_ID},
        "pending": {"candidate": [], "production": []},
        "resolved": {"candidate": [], "production": []},
        "seen": {"candidate": [], "production": []},
        "last_run": {"run_id": "historical-previous-run"},
    }
    windows = {}
    proposal_rows = []
    result_index = 0

    def create_rows(model, count, status_counts, shared_prefix=0):
        nonlocal result_index
        statuses = []
        for status, n in status_counts.items():
            statuses.extend([status] * n)
        rows = []
        for i in range(count):
            if model == "production" and i < shared_prefix:
                symbol = f"SYM{i:03d}"
                open_time = 1_700_000_000_000 + i * 900_000
            else:
                if model == "candidate":
                    unique_no = i
                else:
                    unique_no = 74 + (i - shared_prefix)
                symbol = f"SYM{unique_no:03d}"
                open_time = 1_700_000_000_000 + unique_no * 900_000
            pred = {
                "id": f"{model}-{i:04d}",
                "model": model,
                "symbol": symbol,
                "open_time": open_time,
                "created_at": f"2026-10-01T00:{i % 60:02d}:00+00:00",
                "entry": 100.0,
                "atr": 1.0,
                "signal": "BUY" if model == "candidate" or i % 2 == 0 else "SELL",
                "confidence": 0.81,
                "p_buy": 0.81,
                "p_sell": 0.10,
                "threshold_buy": 0.70,
                "threshold_sell": 0.70,
                "regime_adx": 24.0,
                "regime_atr_pct": 1.2,
            }
            status = statuses[i]
            pred["fixture_status"] = status
            rows.append(pred)
            key = (symbol, open_time)
            windows.setdefault(key, {
                "symbol": symbol,
                "open_time": open_time,
                "candles": [
                    {"timestamp": open_time + j * 900_000, "high": 100.0, "low": 100.0}
                    for j in range(26)
                ],
            })
            gross = {"TP": 3.5, "SL": -2.5, "EXPIRED": 0.0}[status]
            net = gross - 0.12 if status in ("TP", "SL") else -0.12
            proposal_rows.append({
                "prediction_id": pred["id"],
                "observation_key": f'{backfill.EXPERIMENT_ID}:{model}:{symbol}:{open_time}',
                "model": model,
                "symbol": symbol,
                "open_time": open_time,
                "signal": pred["signal"],
                "entry": pred["entry"],
                "atr": pred["atr"],
                "proposed_status": status,
                "outcome_bar": 1,
                "gross_r": gross,
                "net_r": net,
                "candle_window_sha256": "a" * 64,
                "source_created_at": pred["created_at"],
                "resolution_timestamp_policy": "actual UTC reconciliation execution time",
            })
            result_index += 1
        state["pending"][model] = rows
        state["seen"][model] = list(dict.fromkeys(
            f'{row["symbol"]}:{row["open_time"]}' for row in rows
        ))

    create_rows("candidate", 74, {"TP": 12, "SL": 62})
    create_rows("production", 544, {"TP": 280, "SL": 245, "EXPIRED": 19}, shared_prefix=45)

    proposal = {
        "proposal_status": "READY_FOR_REVIEW_NOT_APPLIED",
        "state_or_ledger_modified": False,
        "promotion_authorized": False,
        "proposed_observation_appends": 618,
        "pending_before": {"candidate": 74, "production": 544},
        "pending_after_if_approved": {"candidate": 0, "production": 0},
        "proposed_candidate": {"TP": 12, "SL": 62},
        "proposed_production": {"TP": 280, "SL": 245, "EXPIRED": 19},
        "proposed_total_outcomes": {"TP": 292, "SL": 307, "EXPIRED": 19},
        "proposal_records": proposal_rows,
    }

    def fake_resolver(pred, df):
        status = pred["fixture_status"]
        gross = {"TP": 3.5, "SL": -2.5, "EXPIRED": 0.0}[status]
        return {
            **pred,
            "status": status,
            "outcome_bar": 1 if status != "EXPIRED" else 24,
            "resolved_at": "2026-11-04T12:00:00+00:00",
            "gross_r": gross,
            "net_r": gross - 0.12 if status in ("TP", "SL") else -0.12,
        }
    # The fixture's expiry outcome bar needs match the proposed value.
    for row in proposal_rows:
        if row["proposed_status"] == "EXPIRED":
            row["outcome_bar"] = 24

    return state, proposal, windows, fake_resolver


def test_resolution_plan_matches_all_618_rows_and_573_windows():
    import pandas as pd
    state, proposal, windows, resolver = make_fixture()
    records, metrics = backfill.build_resolution_plan(state, proposal, windows, resolver, pd)
    assert len(records) == 618
    assert metrics["raw_windows_used"] == 573
    assert metrics["candidate"] == {"TP": 12, "SL": 62}
    assert metrics["production"] == {"TP": 280, "SL": 245, "EXPIRED": 19}
    assert metrics["total_outcomes"] == {"TP": 292, "SL": 307, "EXPIRED": 19}


def test_resolution_plan_rejects_proposal_outcome_mismatch():
    import pandas as pd
    state, proposal, windows, resolver = make_fixture()
    proposal["proposal_records"][0]["proposed_status"] = "SL"
    with pytest.raises(backfill.BackfillBundleError, match="mismatch"):
        backfill.build_resolution_plan(state, proposal, windows, resolver, pd)


def test_resolution_plan_requires_exact_pending_id_coverage():
    import pandas as pd
    state, proposal, windows, resolver = make_fixture()
    proposal["proposal_records"].pop()
    with pytest.raises(backfill.BackfillBundleError, match="Expected 618 proposal records"):
        backfill.build_resolution_plan(state, proposal, windows, resolver, pd)


def _write_expected_journal(path: Path, *, actor=None, run_id=None, workflow="workflow_dispatch"):
    rows = [
        {"event": "PHASE2D_RUN", "run_id": str(index)}
        for index in range(backfill.EXPECTED_JOURNAL_ROWS - 3)
    ]
    rows.extend([
        {"event": "F08_CHECKPOINT_EVALUATED", "checkpoint_blocks": 15},
        {"event": "F08_CHECKPOINT_EVALUATED", "checkpoint_blocks": 20},
        {
            "event": "F08_GATE_AMENDMENT",
            "approved_by": actor or backfill.EXPECTED_APPROVAL_ACTOR,
            "approval_run_id": run_id or backfill.EXPECTED_APPROVAL_RUN,
            "workflow_event": workflow,
        },
    ])
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_experiment_journal_accepts_exact_historical_gate_amendment_provenance(tmp_path):
    path = tmp_path / "experiment_ledger.jsonl"
    _write_expected_journal(path)

    result = backfill.validate_experiment_journal(path)

    assert result["rows"] == 98
    assert result["c15_rows"] == 1
    assert result["c20_rows"] == 1
    assert result["c25_or_later_rows"] == 0
    assert result["f08_amendment_rows"] == 1
    assert result["amendment_actor"] == "MrZkull"
    assert result["amendment_run"] == "41720710702"
    assert result["amendment_is_final_row"] is True
    assert result["sha256"] == backfill.sha256_file(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actor": "not-the-owner"},
        {"run_id": "99999999999"},
        {"workflow": "push"},
    ],
)
def test_experiment_journal_rejects_wrong_historical_gate_amendment_provenance(tmp_path, kwargs):
    path = tmp_path / "experiment_ledger.jsonl"
    _write_expected_journal(path, **kwargs)

    with pytest.raises(backfill.BackfillBundleError, match="provenance missing required value"):
        backfill.validate_experiment_journal(path)
