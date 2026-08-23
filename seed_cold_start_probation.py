# seed_cold_start_probation.py — Enforces Cold-Start Probation for New Assets

import os
import json
import time
import base64
import requests
from pathlib import Path

RELIABILITY_FILE = "reliability.json"
COLD_START_SYMBOLS = ["TRUMPUSDT", "PUMPUSDT"]

GH_TOKEN  = os.getenv("GH_PAT_TOKEN", "")
GH_REPO   = os.getenv("GITHUB_REPO", "MrZkull/Crypto_AI_bot")
GH_BRANCH = os.getenv("GITHUB_BRANCH", "main")


def load_reliability() -> dict:
    for p in [Path(RELIABILITY_FILE), Path("data") / RELIABILITY_FILE]:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def save_reliability(data: dict):
    # Save locally
    for p in [Path(RELIABILITY_FILE), Path("data") / RELIABILITY_FILE]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    # Push to GitHub
    if GH_TOKEN and GH_REPO:
        headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        url = f"https://api.github.com/repos/{GH_REPO}/contents/{RELIABILITY_FILE}?ref={GH_BRANCH}"
        try:
            r = requests.get(url, headers=headers, timeout=5)
            sha = r.json().get("sha") if r.ok else None
            payload = {
                "message": "🔒 Seed cold-start probation for TRUMP & PUMP",
                "content": base64.b64encode(json.dumps(data, indent=2).encode('utf-8')).decode('utf-8'),
                "branch": GH_BRANCH
            }
            if sha:
                payload["sha"] = sha
            requests.put(url, headers=headers, json=payload, timeout=8)
            print(f"✓ Pushed updated {RELIABILITY_FILE} to GitHub successfully.")
        except Exception as e:
            print(f"⚠️ GitHub push warning: {e}")


def seed():
    rel = load_reliability()
    now = time.time()
    
    for sym in COLD_START_SYMBOLS:
        rel[sym] = {
            "is_benched": True,
            "benched_at": now,
            "probation_wins": 0,
            "probation_consecutive_losses": 0,
            "normal_consecutive_losses": 0,
            "wins": 0,
            "losses": 0,
            "ghosts": 0,
            "notes": "Cold-start monitoring probation (requires +10% confidence hurdle and 3 wins to graduate)"
        }
        print(f"🔒 {sym} seeded into probation: +10% confidence premium active.")

    save_reliability(rel)


if __name__ == "__main__":
    seed()
