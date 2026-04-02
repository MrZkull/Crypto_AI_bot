web: gunicorn dashboard:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
worker: python live_scanner.py
