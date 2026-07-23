"""
Minimal sanity-check script.

Run from the repo root:
    python examples/run_inference.py "Hello, this is a quick test run."
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.inference import TTSPipeline


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else (
        "Hello, this is but a quick test run to see if everything's working alright."
    )

    pipe = TTSPipeline(
        tts_weights="weights/tts.pt",
        wavtokenizer_weights="weights/wavtokenizer.ckpt",
        wavtokenizer_config="configs/wavtokenizer_config.yaml",
    )
    pipe.generate(text, out_path="output.wav")


if __name__ == "__main__":
    main()
