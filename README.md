# SimpleTTS

Decoder-only transformer TTS, using [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
as the audio codec.

## Setup

```bash
git clone https://github.com/FirstPotatoCoder/SimpleTTS.git
cd SimpleTTS
bash setup.sh
python scripts/download_weights.py
```

## Run

```bash
python examples/run_inference.py "Hello, this is a quick test-run. I'm checking whether the model can handle longer input text without any issues. This sentence is roughly five times the length of the original short test phrase."
```

Or from Python:

```python
from tts.inference import TTSPipeline

pipe = TTSPipeline(
tts_weights="weights/tts.pt",
wavtokenizer_weights="weights/wavtokenizer.ckpt",
wavtokenizer_config="configs/wavtokenizer_config.yaml",
)
pipe.generate("Some text to speak.", out_path="out.wav")
```

## Structure

```
configs/        WavTokenizer model config (yaml)
wavtokenizer/    vendored, trimmed WavTokenizer inference code (encoder/ + decoder/)
tts/               model code: tokenizer.py, model.py, inference.py
scripts/             download_weights.py — pulls weights from Hugging Face
examples/              run_inference.py — minimal end-to-end sanity check
weights/                 (gitignored) downloaded .pt / .ckpt files go here
```

Weights are hosted on Hugging Face, not committed to this repo — run
`scripts/download_weights.py` to fetch them.

## Samples

| Sample   | Text                                      | Download |
| -------- | ------------------------------------------ | -------- |
| Sample 1 | "Hello, this is a quick test-run..."        | [sample_audios/sample_1.wav](sample_audios/sample_1.wav) |
| Sample 2 | "It's been raining all week here..."        | [sample_audios/sample_2.wav](sample_audios/sample_2.wav) |
| Sample 3 | "Artificial intelligence is changing..."    | [sample_audios/sample_3.wav](sample_audios/sample_3.wav) |
| Sample 4 | "We landed in Tokyo just after sunrise..."  | [sample_audios/sample_4.wav](sample_audios/sample_4.wav) |
| Sample 5 | "To brew a good cup of coffee..."           | [sample_audios/sample_5.wav](sample_audios/sample_5.wav) |

### Listen

**Sample 1** — "Hello, this is a quick test-run..."

https://github.com/user-attachments/assets/59c82d85-9c97-4a55-930b-f0d954bc3836

**Sample 2** — "It's been raining all week here..."

https://github.com/user-attachments/assets/e6968cd8-188c-4fad-b1a2-0f89f6a99b1c

**Sample 3** — "Artificial intelligence is changing..."

https://github.com/user-attachments/assets/823611cc-387c-4dde-bd7c-7e4f1df12094

**Sample 4** — "We landed in Tokyo just after sunrise..."

https://github.com/user-attachments/assets/9e0cb6ae-b641-4145-b06f-714127aee110

**Sample 5** — "To brew a good cup of coffee..."

https://github.com/user-attachments/assets/3f51608f-1f88-443c-a753-f555da0ab73e

## Limitations

- The model sometimes hallucinates, generating babbling or garbled output on rare or unseen words — likely due to the limited amount of training data.
- Works best when generating ~3s to 15s of audio, matching the length distribution of its training data.
- No chunking support yet — the current repo only supports clip-by-clip generation, one sample at a time.

## Credits

Audio tokenizer/codec: [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
(vendored under `wavtokenizer/`, trimmed to inference-only code).

Data synthesis: [Kokoro](https://github.com/hexgrad/kokoro), used to generate ~100 hours
of synthetic audio for training this TTS model.

Text phonemization: [espeak-ng](https://github.com/espeak-ng/espeak-ng), used to
convert input text into phonemes.
