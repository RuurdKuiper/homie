#!/usr/bin/env bash
set -euo pipefail

cd /home/homie/projects/homie
exec /home/homie/projects/homie/.venv/bin/python - <<'PY'
from audio_output import ensure_output_sink
ensure_output_sink(log_prefix="[RASPOTIFY]")
PY
