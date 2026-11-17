"""Optional text-to-speech placeholder. Install/configure a TTS engine separately."""
def announce(message: str):
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("Optional dependency missing. Install pyttsx3 to enable speech.") from exc
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()
