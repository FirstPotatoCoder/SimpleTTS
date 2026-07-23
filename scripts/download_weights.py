"""
Pulls model weights from the Hugging Face model repo into ./weights/.

Usage:
    python scripts/download_weights.py --repo-id YourUser/YourRepo
"""
import argparse
import os

from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="AnotherPotatoCoder/English-TTS", help="e.g. AnotherPotatoCoder/English-TTS")
    parser.add_argument("--out-dir", default="weights")
    parser.add_argument("--tts-filename", default="tts.pt")
    parser.add_argument("--wavtokenizer-filename", default="wavtokenizer.ckpt")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for filename in (args.tts_filename, args.wavtokenizer_filename):
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=args.out_dir,
        )
        print(f"✅ {filename} -> {path}")

if __name__ == "__main__":
    main()
