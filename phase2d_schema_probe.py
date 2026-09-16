import hashlib
import json
from pathlib import Path

import joblib


MODEL = Path("pro_crypto_ai_model.pkl")
CANDIDATE = Path("candidate_model.pkl")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(name, path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(path)

    model = joblib.load(path)

    features = list(model.get("all_features", []))

    print("File:", path)
    print("SHA256:", sha256(path))
    print("Feature count:", len(features))
    print("Trained at:", model.get("trained_at"))
    print("Recommended BUY:", model.get("recommended_threshold_buy"))
    print("Recommended SELL:", model.get("recommended_threshold_sell"))

    print("\nFEATURES:")
    for i, feature in enumerate(features, 1):
        print(f"{i:02d}. {feature}")

    print("\nBEST FEATURES:")
    for i, feature in enumerate(model.get("best_features", []), 1):
        print(f"{i:02d}. {feature}")


if __name__ == "__main__":
    inspect("PRODUCTION MODEL", MODEL)

    if CANDIDATE.exists():
        inspect("CANDIDATE MODEL", CANDIDATE)
