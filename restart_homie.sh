#!/usr/bin/env bash
set -euo pipefail

follow_logs=false
if [[ "${1:-}" == "--follow" ]]; then
	follow_logs=true
fi

systemctl --user daemon-reload
systemctl --user restart homie.service
systemctl --user --no-pager --full status homie.service

if [[ -f homie.log ]]; then
	echo
	echo "Recent Homie logs:"
	tail -n 30 homie.log
else
	echo
	echo "homie.log does not exist yet. It will appear after Homie writes output."
fi

if [[ "$follow_logs" == true ]]; then
	echo
	echo "Following homie.log..."
	exec tail -f homie.log
fi