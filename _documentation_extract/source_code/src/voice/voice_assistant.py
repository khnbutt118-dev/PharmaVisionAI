"""Local Kokoro voice preparation and playback for PharmaSight AI."""

import argparse
import sys
import traceback


def prepare(text, output, voice="af_heart", speed=0.96):
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    print("Loading Kokoro voice model...", flush=True)
    pipeline = KPipeline(lang_code="a")
    chunks = []

    print("Generating welcome speech...", flush=True)
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        else:
            audio = np.asarray(audio)
        chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro did not generate any audio.")

    audio_data = np.concatenate(chunks).astype(np.float32)
    sf.write(output, audio_data, 24000)
    print(f"Welcome audio prepared: {output}", flush=True)


def play_file(path):
    import sounddevice as sd
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    print("Playing welcome speech...", flush=True)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    print("Welcome speech completed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=0.96)
    parser.add_argument("--output")
    parser.add_argument("--play-file")
    args = parser.parse_args()

    try:
        if args.play_file:
            play_file(args.play_file)
        elif args.text and args.output:
            prepare(args.text, args.output, args.voice, args.speed)
        else:
            parser.error("Provide --play-file or both --text and --output.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
