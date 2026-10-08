from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


MAGIC = b"CBF08ENC"
VERSION = 1
AAD = b"CryptoBot-AI-F08-canonical-snapshot-v1"

EXPECTED_EXPERIMENT = "PHASE2D-HARDENED-20260924-F08"
EXPECTED_LEDGER_SHA = (
    "493b9353f0d0ac69a4e8d518636bae8e2f8fc5185e580efe5757c8534833c165"
)
EXPECTED_OBSERVATIONS = 9750


def derive_key(passphrase: bytes, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def restore(snapshot: Path, output: Path, passphrase: str) -> None:
    raw = snapshot.read_bytes()

    if raw[:8] != MAGIC:
        raise RuntimeError("Invalid F08 snapshot magic.")

    version = raw[8]

    if version != VERSION:
        raise RuntimeError(
            f"Unsupported F08 snapshot version: {version}"
        )

    iterations = int.from_bytes(
        raw[9:13],
        "big",
    )

    salt = raw[13:29]
    nonce = raw[29:41]
    ciphertext = raw[41:]

    key = derive_key(
        passphrase.encode("utf-8"),
        salt,
        iterations,
    )

    try:
        bundle = AESGCM(key).decrypt(
            nonce,
            ciphertext,
            AAD,
        )
    except Exception as exc:
        raise RuntimeError(
            "F08 snapshot authentication/decryption failed."
        ) from exc

    with zipfile.ZipFile(
        io.BytesIO(bundle),
        "r",
    ) as archive:

        names = set(archive.namelist())

        required = {
            "phase2d_observations_recovered.jsonl",
            "phase2d_state_recovered.json",
            "phase2d_f08_baseline_manifest.json",
        }

        missing = required - names

        if missing:
            raise RuntimeError(
                f"Snapshot missing required files: {sorted(missing)}"
            )

        ledger_bytes = archive.read(
            "phase2d_observations_recovered.jsonl"
        )

        ledger_sha = sha256_bytes(
            ledger_bytes
        )

        if ledger_sha != EXPECTED_LEDGER_SHA:
            raise RuntimeError(
                "Recovered ledger SHA mismatch: "
                f"{ledger_sha}"
            )

        state = json.loads(
            archive.read(
                "phase2d_state_recovered.json"
            )
        )

        manifest = json.loads(
            archive.read(
                "phase2d_f08_baseline_manifest.json"
            )
        )

        experiment = (
            state.get("experiment_definition", {})
            .get("experiment_id")
            or state.get("locked_models", {})
            .get("experiment_id")
        )

        if experiment != EXPECTED_EXPERIMENT:
            raise RuntimeError(
                f"Unexpected experiment ID: {experiment}"
            )

        if manifest.get("experiment_id") != EXPECTED_EXPERIMENT:
            raise RuntimeError(
                "Snapshot manifest experiment ID mismatch."
            )

        if (
            manifest.get("ledger", {})
            .get("observation_count")
            != EXPECTED_OBSERVATIONS
        ):
            raise RuntimeError(
                "Snapshot manifest observation count mismatch."
            )

        if (
            manifest.get("ledger", {})
            .get("sha256")
            != EXPECTED_LEDGER_SHA
        ):
            raise RuntimeError(
                "Snapshot manifest SHA mismatch."
            )

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        (output / "phase2d_observations.jsonl").write_bytes(
            ledger_bytes
        )

        (output / "phase2d_state.json").write_bytes(
            archive.read(
                "phase2d_state_recovered.json"
            )
        )

        (output / "phase2d_f08_baseline_manifest.json").write_bytes(
            archive.read(
                "phase2d_f08_baseline_manifest.json"
            )
        )

        if "recovery_manifest.json" in names:
            (output / "recovery_manifest.json").write_bytes(
                archive.read("recovery_manifest.json")
            )

    print("=" * 70)
    print(" F08 CANONICAL SNAPSHOT RESTORE")
    print("=" * 70)
    print("Snapshot          :", snapshot)
    print("Output            :", output)
    print("PBKDF2 iterations :", iterations)
    print("Experiment        :", EXPECTED_EXPERIMENT)
    print("Observations      :", EXPECTED_OBSERVATIONS)
    print("Ledger SHA256     :", ledger_sha)
    print("")
    print("STATUS            : PASS")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--passphrase-env")
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if args.passphrase_env:
        passphrase = os.environ.get(
            args.passphrase_env,
            "",
        )
        if not passphrase:
            raise SystemExit(
                f"Environment variable {args.passphrase_env} is empty."
            )
    elif args.interactive:
        passphrase = getpass.getpass(
            "F08 snapshot passphrase: "
        )
    else:
        raise SystemExit(
            "Provide --passphrase-env or --interactive."
        )

    restore(
        Path(args.snapshot),
        Path(args.output),
        passphrase,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
