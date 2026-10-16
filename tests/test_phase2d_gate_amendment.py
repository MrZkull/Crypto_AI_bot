from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phase2d_gate_amendment as amendment


EXPERIMENT = amendment.TARGET_EXPERIMENT_ID


def _write_ledger(path: Path, checkpoints=(15, 20), extra_records=()) -> bytes:
    records = [
        {
            "timestamp": "2026-10-15T00:00:00+00:00",
            "event": amendment.CHECKPOINT_EVENT,
            "experiment_id": EXPERIMENT,
            "checkpoint_blocks": checkpoint,
            "decision": "CHECKPOINT_CONTINUE",
            "coverage": {"ready": False},
        }
        for checkpoint in checkpoints
    ]
    records.extend(extra_records)
    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _matching_gate(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "phase2d_promotion_gate.py"
    path.write_bytes(b"reviewed gate test fixture\n")
    monkeypatch.setattr(amendment, "NEW_GATE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_records_once_and_preserves_immutable_c15_c20(tmp_path, monkeypatch):
    ledger = tmp_path / "research_outputs" / "experiment_ledger.jsonl"
    original = _write_ledger(ledger)
    gate = _matching_gate(tmp_path, monkeypatch)

    result = amendment.record_amendment(
        ledger_path=ledger,
        gate_path=gate,
        experiment_id=EXPERIMENT,
        git_ref="refs/heads/main",
        approval_confirmed=True,
        approval_actor="approved-owner",
        approval_run_id="12345",
        approval_event_name="workflow_dispatch",
        authorized_approver="approved-owner",
    )
    assert result == "APPENDED"
    after_first = ledger.read_bytes()
    assert after_first.startswith(original)

    rows = [json.loads(line) for line in after_first.decode("utf-8").splitlines()]
    assert rows[0]["checkpoint_blocks"] == 15
    assert rows[0]["decision"] == "CHECKPOINT_CONTINUE"
    assert rows[1]["checkpoint_blocks"] == 20
    assert rows[1]["decision"] == "CHECKPOINT_CONTINUE"
    assert rows[2]["event"] == amendment.AMENDMENT_EVENT
    assert rows[2]["old_code_sha256"] == amendment.OLD_GATE_SHA256
    assert rows[2]["new_code_sha256"] == amendment.NEW_GATE_SHA256
    assert rows[2]["approval_actor"] == "approved-owner"
    assert rows[2]["approval_run_id"] == "12345"
    assert rows[2]["approval_event_name"] == "workflow_dispatch"
    assert rows[2]["normalization"]["candidate_operation"] == "multiply by 100.0 before volatility-tercile classification"
    assert rows[2]["model_promotion_authorized"] is False

    result_again = amendment.record_amendment(
        ledger_path=ledger,
        gate_path=gate,
        experiment_id=EXPERIMENT,
        git_ref="refs/heads/main",
        authorized_approver="approved-owner",
    )
    assert result_again == "ALREADY_PRESENT"
    assert ledger.read_bytes() == after_first


def test_refuses_to_record_after_c25_and_leaves_ledger_unchanged(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger, checkpoints=(15, 20, 25))
    gate = _matching_gate(tmp_path, monkeypatch)

    with pytest.raises(amendment.AmendmentError, match="already been evaluated"):
        amendment.record_amendment(ledger, gate, EXPERIMENT, "refs/heads/main")
    assert ledger.read_bytes() == original


def test_refuses_missing_c15_or_c20_history(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger, checkpoints=(15,))
    gate = _matching_gate(tmp_path, monkeypatch)

    with pytest.raises(amendment.AmendmentError, match="exactly one immutable C15"):
        amendment.record_amendment(ledger, gate, EXPERIMENT, "refs/heads/main")
    assert ledger.read_bytes() == original


def test_existing_amendment_before_c25_is_idempotent(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    _write_ledger(ledger)
    gate = _matching_gate(tmp_path, monkeypatch)
    assert amendment.record_amendment(
        ledger, gate, EXPERIMENT, "refs/heads/main",
        approval_confirmed=True,
        approval_actor="approved-owner",
        approval_run_id="12345",
        approval_event_name="workflow_dispatch",
        authorized_approver="approved-owner",
    ) == "APPENDED"
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-10-16T00:00:00+00:00",
            "event": amendment.CHECKPOINT_EVENT,
            "experiment_id": EXPERIMENT,
            "checkpoint_blocks": 25,
            "decision": "CHECKPOINT_CONTINUE",
        }, sort_keys=True, separators=(",", ":")) + "\n")
    before_retry = ledger.read_bytes()
    assert amendment.record_amendment(
        ledger,
        gate,
        EXPERIMENT,
        "refs/heads/main",
        authorized_approver="approved-owner",
    ) == "ALREADY_PRESENT"
    assert ledger.read_bytes() == before_retry


def test_refuses_amendment_already_recorded_after_c25(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger, checkpoints=(15, 20, 25))
    gate = _matching_gate(tmp_path, monkeypatch)
    amendment_row = {
        "timestamp": "2026-10-16T00:00:00+00:00",
        "event": amendment.AMENDMENT_EVENT,
        "experiment_id": EXPERIMENT,
        **amendment._payload(),
    }
    appended = json.dumps(amendment_row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    ledger.write_bytes(original + appended)
    before = ledger.read_bytes()
    with pytest.raises(amendment.AmendmentError, match="precedes F08_GATE_AMENDMENT"):
        amendment.record_amendment(ledger, gate, EXPERIMENT, "refs/heads/main")
    assert ledger.read_bytes() == before


def test_wrong_gate_hash_fails_closed_without_writing(tmp_path):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger)
    gate = tmp_path / "wrong_gate.py"
    gate.write_text("not the reviewed gate", encoding="utf-8")

    with pytest.raises(amendment.AmendmentError, match="source hash does not match"):
        amendment.record_amendment(ledger, gate, EXPERIMENT, "refs/heads/main")
    assert ledger.read_bytes() == original


def test_non_main_ref_fails_closed_without_writing(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger)
    gate = _matching_gate(tmp_path, monkeypatch)

    with pytest.raises(amendment.AmendmentError, match="non-main ref"):
        amendment.record_amendment(ledger, gate, EXPERIMENT, "refs/heads/fix/test")
    assert ledger.read_bytes() == original


def test_conflicting_existing_amendment_fails_closed(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    conflict = {
        "timestamp": "2026-10-15T00:00:00+00:00",
        "event": amendment.AMENDMENT_EVENT,
        "experiment_id": EXPERIMENT,
        "reason": "a different reason",
    }
    original = _write_ledger(ledger, extra_records=(conflict,))
    gate = _matching_gate(tmp_path, monkeypatch)

    with pytest.raises(amendment.AmendmentError, match="conflicts with the approved amendment"):
        amendment.record_amendment(ledger, gate, EXPERIMENT, "refs/heads/main")
    assert ledger.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_error"),
    [
        (
            "approval_actor",
            "unapproved-user",
            "approval actor does not match the configured F08 approver",
        ),
        (
            "approval_run_id",
            "not-a-run-id",
            "invalid GitHub workflow run ID",
        ),
    ],
)
def test_existing_amendment_validates_approval_provenance(
    tmp_path, monkeypatch, field, bad_value, expected_error
):
    ledger = tmp_path / "experiment_ledger.jsonl"
    _write_ledger(ledger)
    gate = _matching_gate(tmp_path, monkeypatch)

    result = amendment.record_amendment(
        ledger_path=ledger,
        gate_path=gate,
        experiment_id=EXPERIMENT,
        git_ref="refs/heads/main",
        approval_confirmed=True,
        approval_actor="approved-owner",
        approval_run_id="12345",
        approval_event_name="workflow_dispatch",
        authorized_approver="approved-owner",
    )
    assert result == "APPENDED"

    rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    rows[-1][field] = bad_value
    ledger.write_bytes(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
    )
    before_retry = ledger.read_bytes()

    with pytest.raises(amendment.AmendmentError, match=expected_error):
        amendment.record_amendment(
            ledger_path=ledger,
            gate_path=gate,
            experiment_id=EXPERIMENT,
            git_ref="refs/heads/main",
            authorized_approver="approved-owner",
        )

    assert ledger.read_bytes() == before_retry


def test_refuses_to_append_without_explicit_approval(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger)
    gate = _matching_gate(tmp_path, monkeypatch)

    with pytest.raises(
        amendment.AmendmentError,
        match="Explicit owner/mentor approval is required",
    ):
        amendment.record_amendment(
            ledger_path=ledger,
            gate_path=gate,
            experiment_id=EXPERIMENT,
            git_ref="refs/heads/main",
            approval_confirmed=False,
        )

    assert ledger.read_bytes() == original

def test_refuses_unapproved_actor_even_with_true_flag(tmp_path, monkeypatch):
    ledger = tmp_path / "experiment_ledger.jsonl"
    original = _write_ledger(ledger)
    gate = _matching_gate(tmp_path, monkeypatch)

    with pytest.raises(
        amendment.AmendmentError,
        match="does not match the configured F08 approver",
    ):
        amendment.record_amendment(
            ledger_path=ledger,
            gate_path=gate,
            experiment_id=EXPERIMENT,
            git_ref="refs/heads/main",
            approval_confirmed=True,
            approval_actor="unapproved-user",
            approval_run_id="12345",
            approval_event_name="workflow_dispatch",
            authorized_approver="approved-owner",
        )

    assert ledger.read_bytes() == original


def test_non_target_experiment_is_a_no_op(tmp_path):
    ledger = tmp_path / "does-not-exist.jsonl"
    result = amendment.record_amendment(
        ledger_path=ledger,
        gate_path=tmp_path / "does-not-exist.py",
        experiment_id="ANOTHER-EXPERIMENT",
        git_ref="refs/heads/main",
    )
    assert result == "SKIPPED_NOT_TARGET_EXPERIMENT"
    assert not ledger.exists()


def test_hash_is_independent_of_windows_line_endings(tmp_path):
    lf_path = tmp_path / "gate_lf.py"
    crlf_path = tmp_path / "gate_crlf.py"

    lf_path.write_bytes(b"line one\nline two\n")
    crlf_path.write_bytes(b"line one\r\nline two\r\n")

    assert amendment._sha256(lf_path) == amendment._sha256(crlf_path)

def test_approval_workflow_is_amendment_only_and_guards_state_reset():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "run_phase2d.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert workflow.index(
        "Validate F08 amendment-only dispatch preconditions"
    ) < workflow.index("- name: Restore previous Phase 2D state")

    assert "PHASE2D_ALLOW_FRESH_START" in workflow
    assert "PHASE2D_F08_APPROVER" in workflow
    assert "AMENDMENT-ONLY MODE: prospective validator is skipped; C25 is not evaluated." in workflow
    assert 'if [[ "${{ inputs.approve_f08_gate_amendment || false }}" == "true" ]]; then' in workflow
    assert "Amendment-only approval and historical ledger reconciliation cannot run together." in workflow


def test_expected_gate_hash_matches_canonical_source_text():
    gate_path = Path(__file__).resolve().parents[1] / "phase2d_promotion_gate.py"
    canonical = gate_path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == amendment.NEW_GATE_SHA256