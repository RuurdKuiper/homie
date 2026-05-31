#!/usr/bin/env python3
"""Shared audio sink selection for Homie and spotifyd."""

from __future__ import annotations

import subprocess
import time


selected_output_sink = None


def list_audio_sinks():
	"""List available PulseAudio/PipeWire sinks."""
	try:
		result = subprocess.run(
			["pactl", "list", "short", "sinks"],
			capture_output=True,
			text=True,
			check=False,
		)
	except FileNotFoundError:
		return []

	sinks = []
	for line in result.stdout.splitlines():
		parts = line.split("\t")
		if len(parts) >= 2:
			sinks.append(parts[1].strip())
	return sinks


def choose_output_sink(sinks):
	"""Prefer wired/USB output, then Bluetooth, then the first available sink."""
	if not sinks:
		return None

	def score_sink(sink_name):
		name = sink_name.lower()

		usb_keywords = [
			"usb",
			"dac",
		]
		wired_keywords = [
			"headphones",
			"headset",
			"lineout",
			"line-out",
			"speaker",
			"analog",
		]
		bluetooth_keywords = ["bluez", "bluetooth"]

		if any(keyword in name for keyword in usb_keywords):
			return (0, sink_name)
		if any(keyword in name for keyword in wired_keywords) and not any(
			keyword in name for keyword in bluetooth_keywords
		):
			return (1, sink_name)
		if any(keyword in name for keyword in bluetooth_keywords):
			return (2, sink_name)
		return (3, sink_name)

	return sorted(sinks, key=score_sink)[0]


def ensure_output_sink(log_prefix="[AUDIO]"):
	"""Select the best currently-available output sink and make it default."""
	global selected_output_sink

	sinks = list_audio_sinks()
	chosen_sink = choose_output_sink(sinks)
	if not chosen_sink:
		print(f"{log_prefix} No PulseAudio/PipeWire sinks found; using system default output.")
		selected_output_sink = None
		return None

	if chosen_sink != selected_output_sink:
		print(f"{log_prefix} Selecting output sink: {chosen_sink}")
		subprocess.run(["pactl", "set-default-sink", chosen_sink], check=False)
		time.sleep(0.2)
		selected_output_sink = chosen_sink

	return selected_output_sink