#!/usr/bin/env bash
set -euo pipefail

cd /home/homie/projects/homie
exec /home/homie/projects/homie/.venv/bin/python /home/homie/projects/homie/homie.py >> /home/homie/projects/homie/homie.log 2>&1