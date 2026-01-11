from router import route
from tools import chat, date, weather, spotify

def handle(text: str) -> str:
    intent = route(text)

    if intent.type == "spotify_control":
        return spotify.control(intent.raw_text)

    if intent.type == "spotify_play":
        return spotify.play(intent.raw_text)

    if intent.type == "date":
        return date.handle()

    if intent.type == "weather":
        return weather.handle_with_text(intent.raw_text if intent.raw_text else "")

    if intent.type == "chat_short":
        return chat.short(intent.raw_text)

    if intent.type == "chat_long":
        return chat.long(intent.raw_text)

    return "Sorry, I don’t know how to help with that."
