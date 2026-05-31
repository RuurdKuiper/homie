# Homie

Homie is a Raspberry Pi voice assistant built for always-on use at home. It listens for a wake word, records a spoken command, transcribes it, generates a response with either local or cloud models, and speaks the answer back through Bluetooth or wired audio.

It is designed to run on a Raspberry Pi 5 while still remaining easy to develop over SSH.

## What Homie Can Do

- Wake-word driven interaction with `hey homie`
- Speech-to-text with either local Whisper or OpenAI cloud transcription
- Text generation with either a local llama.cpp model or OpenAI models
- Text-to-speech with either local Piper or OpenAI TTS
- Real-time tool calling through OpenAI function calling
- Wikipedia lookup for stable background knowledge
- Web/news lookup for current information like football results or recent events
- Spotify playback for tracks, artists, albums, and playlists
- Bluetooth speaker output with dynamic fallback to USB-to-AUX or other wired output
- Boot automatically on Raspberry Pi startup via `systemd`

## Current Architecture

Homie currently uses this high-level flow:

1. Vosk listens continuously for the wake word.
2. After wake word detection, the microphone audio is buffered and captured.
3. The spoken request is transcribed with either cloud STT or local Whisper.
4. The text request is sent to either OpenAI or a local LLM.
5. If the LLM decides a tool is needed, Homie executes the tool and feeds the result back to the model.
6. The final answer is spoken with either cloud TTS or local Piper.

## Features And Behavior

### Language

- Wake word: `hey homie`
- Command transcription language: currently configured as Dutch with `STT_LANGUAGE = "nl"`
- LLM responses: same language as the user input when using the cloud LLM path

### Cloud/Local Flexibility

Each major stage can run either locally or through OpenAI:

- `USE_CLOUD_STT`
- `USE_CLOUD_LLM`
- `USE_CLOUD_TTS`

This makes it possible to run:

- fully cloud-backed
- fully local
- hybrid combinations, for example local wake word + cloud LLM + local TTS

### Audio Output Routing

Homie selects output sinks dynamically at playback time.

Priority order:

1. USB / wired audio adapter
2. Bluetooth speaker
3. Any remaining generic system sink

This means:

- Plugging in a USB-to-AUX adapter before the next response should make Homie use it automatically.
- Unplugging it should make Homie fall back to Bluetooth on the next response if Bluetooth is available.
- No restart is required.
- The output sink is checked when playback starts, not only at application boot.

## Project Layout

- `homie.py`: main runtime, wake word handling, STT, TTS, LLM calls, tool routing, sink selection
- `config.py`: configuration flags, model names, API environment variables, audio settings
- `tools/search.py`: Wikipedia and current web/news search helpers
- `tools/spotify.py`: Spotify playback helper
- `tools/weather.py`: weather helper
- `systemd/homie.service`: user service for auto-start on boot
- `.env`: local secrets and API keys, not committed to git

## Requirements

Typical environment for this project:

- Raspberry Pi 5
- Python virtual environment in `.venv`
- PipeWire / PulseAudio available for audio output routing
- Bluetooth support if using a Bluetooth speaker
- Internet access for OpenAI, Wikipedia, web search, and Spotify features

If you are using local models, you also need the relevant model files in `models/`.

## Configuration

Main configuration lives in `config.py`.

Important flags:

```python
USE_CLOUD_STT = True
USE_CLOUD_LLM = True
USE_CLOUD_TTS = True
STT_LANGUAGE = "nl"
WAKE_WORD = "hey homie"
```

Current cloud model defaults:

```python
OPENAI_STT_MODEL = "gpt-4o-mini-transcribe"
OPENAI_LLM_MODEL = "gpt-5.4-mini"
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "alloy"
```

## Environment Variables

Store secrets in `.env`.

Minimum OpenAI setup:

```env
OPENAI_API_KEY=your_openai_api_key
```

Spotify setup:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
SPOTIFY_DEVICE_NAME=Homie
```

Spotify playback on the Raspberry Pi uses a local Raspotify Spotify Connect device named `Homie`.
Homie sets the system default sink immediately before Spotify playback, using the same priority order as TTS:

1. USB / wired audio
2. Bluetooth speaker
3. Any remaining sink

That means Spotify music should follow the same USB-first, Bluetooth-fallback routing as spoken responses.

## Run Manually

Activate the virtual environment and start Homie:

```bash
cd ~/projects/homie
source .venv/bin/activate
python homie.py
```

Or run directly without activating:

```bash
cd ~/projects/homie
./.venv/bin/python homie.py
```

## Development Workflow

After changing Homie code, restart the user service to load the new code immediately:

```bash
cd ~/projects/homie
./restart_homie.sh
```

You do not need to reboot the Raspberry Pi. Restarting the service is enough.

If you want to restart and immediately watch live output in your SSH terminal:

```bash
cd ~/projects/homie
./restart_homie.sh --follow
```

## Start On Boot With systemd

Homie can run automatically at startup while the Raspberry Pi remains fully available over SSH.

Install the user service:

```bash
mkdir -p ~/.config/systemd/user
cp ~/projects/homie/systemd/homie.service ~/.config/systemd/user/homie.service
systemctl --user daemon-reload
systemctl --user enable --now homie.service
loginctl enable-linger homie
```

Useful service commands:

```bash
systemctl --user status homie.service
journalctl --user -u homie.service -f
systemctl --user restart homie.service
systemctl --user stop homie.service
```

Quick restart helper from the project root:

```bash
./restart_homie.sh
```

Show live Homie output after restart:

```bash
./restart_homie.sh --follow
```

Why this approach works well:

- Homie starts automatically at boot
- SSH remains available for development and debugging
- The service runs in the user session, which matches PipeWire, PulseAudio, and Bluetooth audio better than a root system service

## Bluetooth Stability

If the Bluetooth speaker disconnects or idles out after silence, disable the Bluetooth idle timeout.

Edit the Bluetooth config:

```bash
sudo nano /etc/bluetooth/main.conf
```

Under the `[General]` section, add or update this active setting:

```ini
IdleTimeout=0
```

Do not prefix the line with `#`, or it will remain a comment.

Restart Bluetooth after saving:

```bash
sudo systemctl restart bluetooth
```

## Audio Checks And Diagnostics

Inspect current PulseAudio / PipeWire sinks:

```bash
pactl list short sinks
```

Quick sink-selection test from Python:

```bash
./.venv/bin/python - <<'PY'
from homie import list_audio_sinks, choose_output_sink
sinks = list_audio_sinks()
print('SINKS:', sinks)
print('CHOSEN:', choose_output_sink(sinks))
PY
```

If you want to see service logs while testing audio:

```bash
journalctl --user -u homie.service -f
```

## Tools Available To The LLM

When using the OpenAI LLM path, Homie exposes tools to the model so it can fetch or control things proactively.

Current tools:

- `get_time`
- `get_weather`
- `set_reminder`
- `control_device`
- `search_wikipedia`
- `search_web`
- `spotify_play`

The prompt is configured to encourage proactive usage when tools improve accuracy, especially for:

- stable knowledge topics via Wikipedia
- current or fast-changing information via web/news search
- music playback via Spotify

## Spotify Setup

Spotify playback requires API credentials in `.env`.

Create a Spotify app at:

https://developer.spotify.com/dashboard

Add these environment variables to `.env`:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

The same redirect URI must also be configured in the Spotify developer dashboard.

The first real Spotify use may require one-time OAuth authorization.

### Spotify Connect On The Pi

This project includes a Raspotify setup so the Raspberry Pi itself appears in Spotify as a device named `Homie`.

Install and enable it with:

```bash
cd ~/projects/homie
curl -sL https://dtcooper.github.io/raspotify/install.sh | sh
mkdir -p ~/.config/raspotify ~/.config/systemd/user
chmod +x ~/projects/homie/run_raspotify_service.sh ~/projects/homie/run_raspotify_event_hook.sh
cp ~/projects/homie/raspotify/raspotify.conf ~/.config/raspotify/conf
cp ~/projects/homie/systemd/raspotify-homie.service ~/.config/systemd/user/raspotify-homie.service
sudo systemctl disable --now raspotify.service
systemctl --user daemon-reload
systemctl --user enable --now raspotify-homie.service
```

Raspotify uses Spotify Connect discovery, so there is no separate local build step. Once the service is running, the device should appear as `Homie` in Spotify clients on the same network.

Useful Raspotify commands:

```bash
systemctl --user status raspotify-homie.service
journalctl --user -u raspotify-homie.service -f
tail -f ~/projects/homie/raspotify.log
systemctl --user restart raspotify-homie.service
```

## Safe Shutdown

To shut down the Raspberry Pi cleanly:

```bash
sudo shutdown -h now
```

Equivalent command:

```bash
sudo poweroff
```

Schedule shutdown in 5 minutes:

```bash
sudo shutdown -h +5
```

Cancel a scheduled shutdown:

```bash
sudo shutdown -c
```

## Git Hygiene

Local-only files are ignored by `.gitignore`, including:

- `.env`
- `.spotify_cache`
- `.venv/`
- `models/`
- recordings and Python cache files

If secret files were already committed, remove them from git tracking before pushing:

```bash
git rm --cached .env .spotify_cache
git add .gitignore
git commit --amend --no-edit
git push --force-with-lease origin main
```

## Notes

- Cloud-only mode avoids loading local Whisper, Piper, and llama.cpp models unless needed.
- Wake word detection remains local even when other stages use cloud services.
- The local audio stack and available sinks depend on the Pi's current PipeWire / PulseAudio state.
- If audio routing changes while Homie is running, the next playback re-checks the best sink automatically.
