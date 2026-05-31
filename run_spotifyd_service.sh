#!/usr/bin/env bash
set -euo pipefail

cd /home/homie/projects/homie
/home/homie/projects/homie/.venv/bin/python - <<'PY'
from audio_output import ensure_output_sink
ensure_output_sink(log_prefix="[SPOTIFYD]")
PY

exec /home/homie/.local/bin/spotifyd --no-daemon --config-path /home/homie/.config/spotifyd/spotifyd.conf >> /home/homie/projects/homie/spotifyd.log 2>&1