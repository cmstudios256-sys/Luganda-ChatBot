#!/usr/bin/env bash
# Start Yogera. Binds 0.0.0.0 so it is reachable from other devices on your LAN.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then echo "· .env found"; else echo "· no .env (demo mode until you add a key)"; fi
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
