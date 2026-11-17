import re
from pathlib import Path

path = Path("src/camera/pharmasight_live.py")
source = path.read_text(encoding="utf-8")

new_functions = r'''
def prepare_welcome_audio():
    """Generate the welcome audio before the webcam opens."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    voice_python = root / ".venv_voice" / "Scripts" / "python.exe"
    voice_script = root / "src" / "voice" / "voice_assistant.py"
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    audio_file = log_dir / "welcome.wav"

    if not voice_python.is_file() or not voice_script.is_file():
        print("Kokoro voice environment or script was not found.")
        return None

    command = [
        str(voice_python), str(voice_script),
        "--text",
        "Hello, Hassan! Welcome to your Pharma Vision AI. How can I help you today?",
        "--voice", "af_heart",
        "--speed", "0.96",
        "--output", str(audio_file)
    ]

    try:
        with open(log_dir / "voice_assistant.log", "a", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=str(root), stdout=log,
                stderr=subprocess.STDOUT, check=False
            )
        if result.returncode == 0 and audio_file.is_file():
            print("Welcome audio is ready.")
            return audio_file
        print("Welcome audio preparation failed. Check logs/voice_assistant.log.")
    except Exception as exc:
        print(f"Could not prepare Kokoro welcome audio: {exc}")

    return None


def play_welcome_audio(audio_file):
    """Start playback of the already-generated welcome audio."""
    import subprocess
    from pathlib import Path

    if audio_file is None:
        return None

    root = Path(__file__).resolve().parents[2]
    voice_python = root / ".venv_voice" / "Scripts" / "python.exe"
    voice_script = root / "src" / "voice" / "voice_assistant.py"
    log_file = open(root / "logs" / "voice_assistant.log", "a", encoding="utf-8")

    try:
        process = subprocess.Popen(
            [str(voice_python), str(voice_script), "--play-file", str(audio_file)],
            cwd=str(root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        return process
    except Exception as exc:
        print(f"Kokoro playback could not start: {exc}")
        return None
    finally:
        log_file.close()

'''

pattern = r'(?s)def speak_welcome\(\):.*?(?=\ndef draw_welcome_overlay\()'
match = re.search(pattern, source)
if not match:
    raise RuntimeError("Could not locate the existing speak_welcome function.")

source = source[:match.start()] + new_functions + source[match.end():]

old_start = '''        print("All components loaded. Opening webcam...")

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)'''
new_start = '''        print("All components loaded.")
        print("Preparing welcome voice before opening webcam...")
        welcome_audio = prepare_welcome_audio()
        print("Opening webcam...")

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)'''

if old_start not in source:
    raise RuntimeError("Could not locate the expected webcam startup section.")

source = source.replace(old_start, new_start, 1)

old_welcome = '''        # One-time startup welcome.
        welcome_started = time.perf_counter()
        speak_welcome()

        start = time.perf_counter()'''
new_welcome = '''        # Start the pre-generated welcome immediately after the webcam opens.
        welcome_started = time.perf_counter()
        welcome_process = play_welcome_audio(welcome_audio)

        start = time.perf_counter()'''

if old_welcome not in source:
    raise RuntimeError("Could not locate the expected welcome startup section.")

source = source.replace(old_welcome, new_welcome, 1)

old_overlay = '''            if time.perf_counter() - welcome_started < 7.0:
                draw_welcome_overlay(display, time.perf_counter() - welcome_started)'''
new_overlay = '''            # Keep the greeting visible until the welcome voice finishes.
            welcome_is_playing = (
                welcome_process is not None and welcome_process.poll() is None
            )
            welcome_fallback = (
                welcome_process is None
                and time.perf_counter() - welcome_started < 7.0
            )
            if welcome_is_playing or welcome_fallback:
                draw_welcome_overlay(display, time.perf_counter() - welcome_started)'''

if old_overlay not in source:
    raise RuntimeError("Could not locate the expected welcome overlay timing section.")

source = source.replace(old_overlay, new_overlay, 1)
path.write_text(source, encoding="utf-8")
print("Both files patched successfully.")
