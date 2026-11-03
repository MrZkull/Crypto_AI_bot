from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase2d_f08_archive_restore import (
    ArchiveError,
    MAGIC,
    VERSION,
    _decrypt_bytes,
    _encrypt_bytes,
    encrypt_archive,
    _validate_archive_payloads,
    _merge_validation_stats,
    _validate_project_state,
    _print_summary,
    EXPECTED_FILES,
)


PASSPHRASE = "test-only-long-random-passphrase-value-1234567890"


class TestF08ArchiveEnvelope(unittest.TestCase):
    def test_authenticated_round_trip(self):
        payload = b"small test archive payload"
        encrypted = _encrypt_bytes(payload, PASSPHRASE)
        self.assertTrue(encrypted.startswith(MAGIC))
        self.assertEqual(encrypted[8], VERSION)
        self.assertEqual(_decrypt_bytes(encrypted, PASSPHRASE), payload)

    def test_wrong_passphrase_fails_closed(self):
        encrypted = _encrypt_bytes(b"payload", PASSPHRASE)
        with self.assertRaises(ArchiveError):
            _decrypt_bytes(encrypted, "a-different-but-long-passphrase-123456")

    def test_tampered_ciphertext_fails_authentication(self):
        encrypted = bytearray(_encrypt_bytes(b"payload", PASSPHRASE))
        encrypted[-1] ^= 0x01
        with self.assertRaises(ArchiveError):
            _decrypt_bytes(bytes(encrypted), PASSPHRASE)

    def test_wrong_magic_is_rejected(self):
        encrypted = bytearray(_encrypt_bytes(b"payload", PASSPHRASE))
        encrypted[0] ^= 0x01
        with self.assertRaises(ArchiveError):
            _decrypt_bytes(bytes(encrypted), PASSPHRASE)

    def test_truncated_envelope_is_rejected(self):
        with self.assertRaises(ArchiveError):
            _decrypt_bytes(b"short", PASSPHRASE)

    def test_unsupported_kdf_iteration_count_is_rejected(self):
        encrypted = bytearray(_encrypt_bytes(b"payload", PASSPHRASE))
        encrypted[9:13] = (100).to_bytes(4, "big")
        with self.assertRaises(ArchiveError):
            _decrypt_bytes(bytes(encrypted), PASSPHRASE)

    def test_incomplete_archive_member_set_is_rejected(self):
        with self.assertRaises(ArchiveError):
            _validate_archive_payloads({})

    def test_bad_internal_payload_is_rejected(self):
        payloads = {name: b"x" for name in EXPECTED_FILES}
        with self.assertRaises(ArchiveError):
            _validate_archive_payloads(payloads)

    def test_encrypt_refuses_unpinned_plaintext_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "wrong.zip"
            destination = Path(temp) / "wrong.enc"
            source.write_bytes(b"not the authoritative ZIP")
            with self.assertRaises(ArchiveError):
                encrypt_archive(source, destination, PASSPHRASE)
            self.assertFalse(destination.exists())

    def test_archive_and_guard_metrics_are_merged_using_matching_names(self):
        archive_stats = {
            "observation_count": 16208,
            "candidate_count": 2959,
            "production_count": 13249,
            "event_count": 3,
            "journal_count": 97,
            "checkpoint_ids": [15, 20],
            "pending_candidate": 74,
            "pending_production": 544,
        }
        guard_stats = {
            "observation_count": 16208,
            "candidate_observation_count": 2959,
            "production_observation_count": 13249,
            "unique_utc_blocks": 23,
            "sha256": "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9",
        }

        merged = _merge_validation_stats(archive_stats, guard_stats)
        self.assertEqual(merged["candidate_count"], 2959)
        self.assertEqual(merged["production_count"], 13249)
        self.assertEqual(merged["ledger_sha256"], guard_stats["sha256"])

    def test_archive_and_guard_count_disagreement_fails_closed(self):
        archive_stats = {
            "observation_count": 16208,
            "candidate_count": 2959,
            "production_count": 13249,
        }
        guard_stats = {
            "observation_count": 16208,
            "candidate_observation_count": 2958,
            "production_observation_count": 13250,
        }
        with self.assertRaises(ArchiveError):
            _merge_validation_stats(archive_stats, guard_stats)

    def test_summary_accepts_merged_archive_and_guard_stats(self):
        from contextlib import redirect_stdout
        from io import StringIO

        stats = _merge_validation_stats(
            {
                "observation_count": 16208,
                "candidate_count": 2959,
                "production_count": 13249,
                "event_count": 3,
                "journal_count": 97,
                "checkpoint_ids": [15, 20],
                "pending_candidate": 74,
                "pending_production": 544,
            },
            {
                "observation_count": 16208,
                "candidate_observation_count": 2959,
                "production_observation_count": 13249,
                "unique_utc_blocks": 23,
                "sha256": "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9",
            },
        )
        output = StringIO()
        with redirect_stdout(output):
            _print_summary(stats)
        rendered = output.getvalue()
        self.assertIn("Candidate: 2959", rendered)
        self.assertIn("Production: 13249", rendered)
        self.assertIn("Ledger SHA256:", rendered)

    def test_merge_rejects_missing_guard_metrics(self):
        archive_stats = {
            "observation_count": 16208,
            "candidate_count": 2959,
            "production_count": 13249,
            "event_count": 3,
            "journal_count": 97,
            "checkpoint_ids": [15, 20],
            "pending_candidate": 74,
            "pending_production": 544,
        }
        with self.assertRaises(ArchiveError):
            _merge_validation_stats(archive_stats, {"observation_count": 16208})

    def test_project_state_integrity_validation_is_fail_closed(self):
        from types import ModuleType
        from unittest.mock import patch

        fake_validator = ModuleType("phase2d_prospective_validator")
        def reject_state(_state):
            raise ValueError("fixture state corruption")
        fake_validator.validate_state_integrity = reject_state

        with patch.dict("sys.modules", {"phase2d_prospective_validator": fake_validator}):
            with self.assertRaisesRegex(ArchiveError, "fixture state corruption"):
                _validate_project_state({"schema_version": 6})


if __name__ == "__main__":
    unittest.main()
