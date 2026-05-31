#!/usr/bin/env python3
"""Spotify playback helper for Homie tools."""

from __future__ import annotations

import socket
import sys

from audio_output import ensure_output_sink
import config


SPOTIFY_SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"


def _get_spotify_client():
	client_id = getattr(config, "SPOTIFY_CLIENT_ID", "")
	client_secret = getattr(config, "SPOTIFY_CLIENT_SECRET", "")
	redirect_uri = getattr(config, "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
	cache_path = getattr(config, "SPOTIFY_CACHE_PATH", ".spotify_cache")

	if not client_id or not client_secret:
		return None, (
			"Spotify is not configured yet. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to .env "
			"from https://developer.spotify.com/dashboard."
		)

	try:
		import spotipy
		from spotipy.oauth2 import SpotifyOAuth
	except ImportError:
		return None, "Spotify control requires the 'spotipy' package. Install it with: pip install spotipy"

	try:
		auth_manager = SpotifyOAuth(
			client_id=client_id,
			client_secret=client_secret,
			redirect_uri=redirect_uri,
			scope=SPOTIFY_SCOPES,
			cache_path=cache_path,
			open_browser=False,
		)
		return spotipy.Spotify(auth_manager=auth_manager), None
	except Exception as exc:
		print(f"[ERROR] Spotify authentication failed: {exc}", file=sys.stderr)
		return None, "Spotify authentication failed. Complete the one-time OAuth setup after adding your credentials."


def _normalized(value: str) -> str:
	return " ".join((value or "").strip().lower().split())


def _preferred_device_label() -> str:
	preferred_device_id = getattr(config, "SPOTIFY_DEVICE_ID", "").strip()
	preferred_device_name = getattr(config, "SPOTIFY_DEVICE_NAME", "").strip()
	return preferred_device_name or preferred_device_id


def _pick_device_from_list(devices):
	preferred_device_id = getattr(config, "SPOTIFY_DEVICE_ID", "").strip()
	preferred_device_name = getattr(config, "SPOTIFY_DEVICE_NAME", "").strip()
	hostname = socket.gethostname().strip()

	if preferred_device_id:
		for device in devices:
			if device.get("id") == preferred_device_id:
				return device

	if preferred_device_name:
		normalized_target = _normalized(preferred_device_name)
		for device in devices:
			if _normalized(device.get("name", "")) == normalized_target:
				return device
		for device in devices:
			if normalized_target in _normalized(device.get("name", "")):
				return device

	if hostname:
		normalized_hostname = _normalized(hostname)
		for device in devices:
			if normalized_hostname == _normalized(device.get("name", "")):
				return device

	return next((device for device in devices if device.get("is_active")), devices[0])


def _pick_device(sp):
	devices = sp.devices().get("devices", [])
	if not devices:
		return None

	device = _pick_device_from_list(devices)
	preferred_label = _preferred_device_label()
	if preferred_label and not any(
		candidate.get("id") == device.get("id") for candidate in devices
	):
		return None
	return device


def _ensure_target_device(sp, device):
	device_id = (device or {}).get("id")
	if not device_id:
		return None
	if not device.get("is_active"):
		sp.transfer_playback(device_id=device_id, force_play=False)
	return device_id


def spotify_play(track_or_artist: str, context: str = "track") -> str:
	"""Play a track, artist, album, or playlist on Spotify."""
	query = track_or_artist.strip()
	if not query:
		return "Spotify play needs a song, artist, album, or playlist name."

	sp, error = _get_spotify_client()
	if error:
		return error

	try:
		ensure_output_sink(log_prefix="[SPOTIFY]")
		devices = sp.devices().get("devices", [])
		device = _pick_device_from_list(devices) if devices else None
		if not device:
			return "Spotify is connected, but no playback device is available. Open Spotify on one of your devices first."

		preferred_label = _preferred_device_label()
		if preferred_label:
			preferred_matches = [
				candidate for candidate in devices
				if candidate.get("id") == getattr(config, "SPOTIFY_DEVICE_ID", "").strip()
				or _normalized(candidate.get("name", "")) == _normalized(getattr(config, "SPOTIFY_DEVICE_NAME", "").strip())
				or (
					getattr(config, "SPOTIFY_DEVICE_NAME", "").strip()
					and _normalized(getattr(config, "SPOTIFY_DEVICE_NAME", "").strip()) in _normalized(candidate.get("name", ""))
				)
			]
			if not preferred_matches:
				return f"The preferred Spotify device '{preferred_label}' is not available right now. Open Spotify on that device first."
			device = preferred_matches[0]

		device_id = _ensure_target_device(sp, device)
		search_type = context if context in {"track", "artist", "playlist", "album"} else "track"
		result = sp.search(q=query, type=search_type, limit=1)

		if search_type == "track":
			items = result.get("tracks", {}).get("items", [])
			if not items:
				return f"I could not find a Spotify track for '{query}'."
			track = items[0]
			sp.start_playback(device_id=device_id, uris=[track["uri"]])
			artist_names = ", ".join(artist["name"] for artist in track.get("artists", []))
			return f"Now playing '{track['name']}' by {artist_names} on Spotify on {device.get('name', 'the selected device')}."

		if search_type == "artist":
			items = result.get("artists", {}).get("items", [])
			if not items:
				return f"I could not find a Spotify artist for '{query}'."
			artist = items[0]
			top_tracks = sp.artist_top_tracks(artist["id"]).get("tracks", [])
			if not top_tracks:
				return f"I found {artist['name']} on Spotify, but no playable tracks were available."
			top_uris = [track["uri"] for track in top_tracks[:5]]
			sp.start_playback(device_id=device_id, uris=top_uris)
			return f"Now playing top tracks from {artist['name']} on Spotify on {device.get('name', 'the selected device')}."

		if search_type == "album":
			items = result.get("albums", {}).get("items", [])
			if not items:
				return f"I could not find a Spotify album for '{query}'."
			album = items[0]
			sp.start_playback(device_id=device_id, context_uri=album["uri"])
			artist_names = ", ".join(artist["name"] for artist in album.get("artists", []))
			return f"Now playing the album '{album['name']}' by {artist_names} on Spotify on {device.get('name', 'the selected device')}."

		items = result.get("playlists", {}).get("items", [])
		if not items:
			return f"I could not find a Spotify playlist for '{query}'."
		playlist = items[0]
		sp.start_playback(device_id=device_id, context_uri=playlist["uri"])
		owner = playlist.get("owner", {}).get("display_name", "Spotify")
		return f"Now playing the playlist '{playlist['name']}' by {owner} on Spotify on {device.get('name', 'the selected device')}."
	except Exception as exc:
		print(f"[ERROR] Spotify playback failed for '{query}': {exc}", file=sys.stderr)
		return f"Spotify playback failed for '{query}'."
