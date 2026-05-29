#!/usr/bin/env python3
"""Spotify playback helper for Homie tools."""

from __future__ import annotations

import sys

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


def _pick_device(sp):
	devices = sp.devices().get("devices", [])
	if not devices:
		return None
	return next((device for device in devices if device.get("is_active")), devices[0])


def spotify_play(track_or_artist: str, context: str = "track") -> str:
	"""Play a track, artist, album, or playlist on Spotify."""
	query = track_or_artist.strip()
	if not query:
		return "Spotify play needs a song, artist, album, or playlist name."

	sp, error = _get_spotify_client()
	if error:
		return error

	try:
		device = _pick_device(sp)
		if not device:
			return "Spotify is connected, but no playback device is available. Open Spotify on one of your devices first."

		device_id = device.get("id")
		search_type = context if context in {"track", "artist", "playlist", "album"} else "track"
		result = sp.search(q=query, type=search_type, limit=1)

		if search_type == "track":
			items = result.get("tracks", {}).get("items", [])
			if not items:
				return f"I could not find a Spotify track for '{query}'."
			track = items[0]
			sp.start_playback(device_id=device_id, uris=[track["uri"]])
			artist_names = ", ".join(artist["name"] for artist in track.get("artists", []))
			return f"Now playing '{track['name']}' by {artist_names} on Spotify."

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
			return f"Now playing top tracks from {artist['name']} on Spotify."

		if search_type == "album":
			items = result.get("albums", {}).get("items", [])
			if not items:
				return f"I could not find a Spotify album for '{query}'."
			album = items[0]
			sp.start_playback(device_id=device_id, context_uri=album["uri"])
			artist_names = ", ".join(artist["name"] for artist in album.get("artists", []))
			return f"Now playing the album '{album['name']}' by {artist_names} on Spotify."

		items = result.get("playlists", {}).get("items", [])
		if not items:
			return f"I could not find a Spotify playlist for '{query}'."
		playlist = items[0]
		sp.start_playback(device_id=device_id, context_uri=playlist["uri"])
		owner = playlist.get("owner", {}).get("display_name", "Spotify")
		return f"Now playing the playlist '{playlist['name']}' by {owner} on Spotify."
	except Exception as exc:
		print(f"[ERROR] Spotify playback failed for '{query}': {exc}", file=sys.stderr)
		return f"Spotify playback failed for '{query}'."
