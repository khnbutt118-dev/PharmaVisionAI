from pathlib import Path
import sys
import py_compile

path = Path(r"src/camera/pharmasight_live.py")
original = path.read_text(encoding="utf-8")
updated = original

def replace_once(old, new, description):
    global updated
    count = updated.count(old)
    if count != 1:
        raise RuntimeError(
            f"{description}: expected exactly one match, found {count}"
        )
    updated = updated.replace(old, new, 1)

replace_once(
    "import time\nimport math\n",
    "import time\nimport math\nimport subprocess\n",
    "subprocess import"
)

replace_once(
    '        self.last_ppe_status = "PPE mode ready"\n',
    '''        self.last_ppe_status = "PPE mode ready"
        self.ppe_voice_last_state = None
        self.ppe_voice_process = None
''',
    "PPE voice state initialization"
)

replace_once(
    '        self.mode = mode\n',
    '''        self.mode = mode
        if mode == "PPE":
            self.ppe_voice_last_state = None
''',
    "PPE mode transition reset"
)

old_method = '''    def process_ppe(self, frame):
        if self.frame_number % 2 == 0 or self.cached_ppe_frame is None:
            annotated, info = self.ppe.process(frame.copy(), draw=True)
            self.cached_ppe_frame = annotated
            self.last_ppe_status = str(info.get("status", "UNCERTAIN"))
        out = self.cached_ppe_frame.copy()
        cv2.putText(out, "PPE detection is a research awareness aid, not validated access control",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (255, 220, 120), 1, cv2.LINE_AA)
        self.status = f"PPE status: {self.last_ppe_status}"
        return out
'''

new_method = '''    def speak_ppe_event(self, event):
        # Play pre-generated Kokoro audio without blocking webcam processing.
        audio_files = {
            "NO_MASK": ROOT / "logs" / "ppe_voice" / "please_wear_mask.wav",
            "COMPLIANT": ROOT / "logs" / "ppe_voice" / "great_job_mask.wav",
        }

        audio_path = audio_files.get(event)
        voice_python = ROOT / ".venv_voice" / "Scripts" / "python.exe"
        voice_script = ROOT / "src" / "voice" / "voice_assistant.py"

        if not audio_path or not audio_path.is_file():
            return
        if not voice_python.is_file() or not voice_script.is_file():
            return

        # Do not overlap announcements.
        if (self.ppe_voice_process is not None
                and self.ppe_voice_process.poll() is None):
            return

        try:
            self.ppe_voice_process = subprocess.Popen(
                [
                    str(voice_python),
                    str(voice_script),
                    "--play-file",
                    str(audio_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"PPE voice playback could not start: {exc}")

    def process_ppe(self, frame):
        if self.frame_number % 2 == 0 or self.cached_ppe_frame is None:
            annotated, info = self.ppe.process(frame.copy(), draw=True)
            self.cached_ppe_frame = annotated
            self.last_ppe_status = str(info.get("status", "UNCERTAIN"))

            status = self.last_ppe_status

            if status == "NOT_COMPLIANT":
                # Speak only for a detected no-mask condition.
                # Incorrectly worn masks do not receive appreciation.
                if (info.get("has_no_mask", False)
                        and not info.get("has_incorrect", False)):
                    if self.ppe_voice_last_state != "NO_MASK":
                        self.ppe_voice_last_state = "NO_MASK"
                        self.speak_ppe_event("NO_MASK")
                else:
                    self.ppe_voice_last_state = None

            elif status == "COMPLIANT":
                if self.ppe_voice_last_state != "COMPLIANT":
                    self.ppe_voice_last_state = "COMPLIANT"
                    self.speak_ppe_event("COMPLIANT")

            else:
                # UNCERTAIN and NO_FACE never produce voice announcements.
                self.ppe_voice_last_state = None

        out = self.cached_ppe_frame.copy()
        cv2.putText(out, "PPE detection is a research awareness aid, not validated access control",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (255, 220, 120), 1, cv2.LINE_AA)
        self.status = f"PPE status: {self.last_ppe_status}"
        return out
'''

replace_once(old_method, new_method, "PPE processing method")

path.write_text(updated, encoding="utf-8")

try:
    py_compile.compile(str(path), doraise=True)
    print("PATCH SUCCESSFUL")
    print("Python syntax check: PASSED")
except Exception:
    path.write_text(original, encoding="utf-8")
    print("PATCH FAILED: original file restored")
    raise
