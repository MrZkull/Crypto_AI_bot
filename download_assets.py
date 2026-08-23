# download_assets.py — Resilient chunked streaming for Render builds
import os
import time
import requests

MODEL_URL = "https://github.com/MrZkull/Crypto_AI_bot/releases/download/v2.0/pro_crypto_ai_model.pkl"
OUTPUT_FILE = "pro_crypto_ai_model.pkl"
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks


def download_with_retry(url: str, dest: str, max_retries: int = 5):
    headers = {"User-Agent": "CryptoBot-Deployer/1.0"}
    for attempt in range(1, max_retries + 1):
        try:
            print(f"📥 Downloading model from release (attempt {attempt}/{max_retries})...")
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_bytes = int(r.headers.get("content-length", 0))
                downloaded = 0
                
                with open(dest + ".tmp", "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_bytes:
                                print(f"  Progress: {downloaded / (1024*1024):.1f}MB / {total_bytes / (1024*1024):.1f}MB ({downloaded / total_bytes * 100:.1f}%)", end="\r")

            os.replace(dest + ".tmp", dest)
            print(f"\n✅ Successfully downloaded {dest} ({os.path.getsize(dest) / (1024*1024):.1f} MB)")
            return
        except Exception as e:
            print(f"\n⚠️ Download attempt {attempt} failed: {e}")
            if os.path.exists(dest + ".tmp"):
                os.remove(dest + ".tmp")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"🚨 Failed to download {dest} after {max_retries} attempts.")


if __name__ == "__main__":
    download_with_retry(MODEL_URL, OUTPUT_FILE)
