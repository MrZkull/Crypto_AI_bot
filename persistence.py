# persistence.py
# Solves the "Cloud Amnesia" problem permanently.
# Stores all bot state (trades, history, signals) in your GitHub repo
# so Render restarts NEVER wipe your data again.
#
# How it works:
#   - GitHub Actions (bot) writes JSON → commits to GitHub repo
#   - Render (dashboard) reads JSON → fetches from GitHub repo
#   - Both always have the same data regardless of restarts

import os, json, base64, requests, logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GH_PAT_TOKEN", "")
GITHUB_REPO  = os.getenv("GITHUB_REPO",  "Elliot14R/Crypto_AI_bot")
GITHUB_BRANCH= os.getenv("GITHUB_BRANCH","main")

GITHUB_API   = "https://api.github.com"
HEADERS      = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Files that get persisted to GitHub
PERSISTENT_FILES = [
    "trades.json",
    "trade_history.json",
    "signals.json",
    "scan_mode.json",
]


def _get_file_sha(filename: str) -> str | None:
    """Get current SHA of a file in GitHub (needed to update it)."""
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/{filename}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json().get("sha")
    except Exception:
        pass
    return None


def save_to_github(filename: str, data: dict | list) -> bool:
    """
    Save JSON data to GitHub repo under /data/ folder.
    Creates the file if it doesn't exist, updates it if it does.
    """
    if not GITHUB_TOKEN:
        return False

    url     = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/{filename}"
    content = base64.b64encode(
        json.dumps(data, indent=2, default=str).encode()
    ).decode()
    sha     = _get_file_sha(filename)

    payload = {
        "message": f"bot: update {filename} [{datetime.now(timezone.utc).strftime('%H:%M UTC')}]",
        "content": content,
        "branch":  GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    try:
        r = requests.put(url, headers=HEADERS, json=payload, timeout=15)
        if r.status_code in (200, 201):
            return True
        log.warning(f"GitHub save failed for {filename}: {r.status_code} {r.text[:100]}")
    except Exception as e:
        log.warning(f"GitHub save error for {filename}: {e}")
    return False


def load_from_github(filename: str, default):
    """
    Load JSON data from GitHub repo.
    Falls back to local file, then to default if both fail.
    """
    # Try GitHub first
    if GITHUB_TOKEN:
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/{filename}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                content = r.json().get("content", "")
                decoded = base64.b64decode(content).decode()
                return json.loads(decoded)
        except Exception as e:
            log.debug(f"GitHub load failed for {filename}: {e}")

    # Fallback to local file
    try:
        p = Path(filename)
        if p.exists():
            with open(p) as f:
                return json.load(f)
    except Exception:
        pass

    return default


def load_json(path: str, default):
    """Drop-in replacement for the existing load_json function."""
    return load_from_github(path, default)


import os, json, base64, requests, time, threading
import logging

log = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO  = os.getenv("GITHUB_REPO")  # Format: "username/repo"
BRANCH       = "main"

# 🚦 THE FIX: Create a "Lock" so only one thread can talk to GitHub at a time
_save_lock = threading.Lock()

def save_json(filename, data):
    """Thread-safe save to GitHub with a smart retry loop."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Convert data to base64 for GitHub
    content_str = json.dumps(data, indent=2, default=str)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
    
    # 🚦 Tell the thread to wait its turn in line
    with _save_lock:
        # Try up to 5 times to get the correct SHA and save
        for attempt in range(5):
            try:
                # 1. Fetch the absolute newest SHA from GitHub
                r_get = requests.get(url, headers=headers, timeout=10)
                sha = r_get.json().get("sha") if r_get.ok else None
                
                # 2. Build the payload
                payload = {
                    "message": f"Auto-update {filename}",
                    "content": content_b64,
                    "branch": BRANCH
                }
                if sha:
                    payload["sha"] = sha
                    
                # 3. Push the save
                r_put = requests.put(url, headers=headers, json=payload, timeout=15)
                
                if r_put.ok:
                    return  # Success! Exit the function.
                    
                elif r_put.status_code == 409:
                    log.warning(f"GitHub 409 Conflict for {filename} (Attempt {attempt+1}/5). Retrying in 2s...")
                    time.sleep(2)  # Wait 2 seconds for the file to settle, then try again
                    continue
                    
                else:
                    log.error(f"GitHub save failed: {r_put.status_code} {r_put.text}")
                    break # A different error happened, stop trying
                    
            except Exception as e:
                log.error(f"GitHub request error for {filename}: {e}")
                break


def sync_all_to_github():
    """
    Push all local JSON files to GitHub at once.
    Call this at bot startup to ensure GitHub has latest state.
    """
    synced = 0
    for filename in PERSISTENT_FILES:
        p = Path(filename)
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                if save_to_github(filename, data):
                    synced += 1
            except Exception:
                pass
    log.info(f"  Synced {synced}/{len(PERSISTENT_FILES)} files to GitHub")
    return synced


def pull_all_from_github():
    """
    Pull all JSON files from GitHub to local disk.
    Call this at Render startup to restore state after restart.
    """
    pulled = 0
    for filename in PERSISTENT_FILES:
        data = load_from_github(filename, None)
        if data is not None:
            try:
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                pulled += 1
                log.info(f"  Restored {filename} from GitHub ({len(data) if isinstance(data, list) else 'dict'})")
            except Exception as e:
                log.warning(f"  Failed to restore {filename}: {e}")
    log.info(f"  Restored {pulled}/{len(PERSISTENT_FILES)} files from GitHub")
    return pulled


def get_stats() -> dict:
    """Return stats about current persistence state."""
    stats = {}
    for filename in PERSISTENT_FILES:
        data = load_from_github(filename, None)
        if data is None:
            stats[filename] = "missing"
        elif isinstance(data, list):
            stats[filename] = f"{len(data)} records"
        elif isinstance(data, dict):
            stats[filename] = f"{len(data)} keys"
    return stats
