#!/usr/bin/env bash
set -euo pipefail

cd /home/homie/projects/homie
set -a
source /home/homie/.config/raspotify/conf
set +a

/home/homie/projects/homie/.venv/bin/python - <<'PY'
from audio_output import ensure_output_sink
ensure_output_sink(log_prefix="[RASPOTIFY]")
PY

exec /usr/bin/librespot >> /home/homie/projects/homie/raspotify.log 2>&1
