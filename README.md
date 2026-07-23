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

https://github.com/user-attachments/assets/sample_1.wav

| Sample   | Audio                                                    |
| -------- | -------------------------------------------------------- |
| Sample 1 | [sample_audios/sample_1.wav](sample_audios/sample_1.wav) |
| Sample 2 | [sample_audios/sample_2.wav](sample_audios/sample_2.wav) |
| Sample 3 | [sample_audios/sample_3.wav](sample_audios/sample_3.wav) |
| Sample 4 | [sample_audios/sample_4.wav](sample_audios/sample_4.wav) |
| Sample 5 | [sample_audios/sample_5.wav](sample_audios/sample_5.wav) |

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
