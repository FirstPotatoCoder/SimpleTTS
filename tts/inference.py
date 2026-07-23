"""
End-to-end inference: text -> phoneme ids -> audio codes -> waveform.

Usage:
    from tts.inference import TTSPipeline

    pipe = TTSPipeline(
        tts_weights="weights/tts_fp16.pt",
        wavtokenizer_weights="weights/wavtokenizer_inference.ckpt",
        wavtokenizer_config="configs/wavtokenizer_config.yaml",
    )
    audio = pipe.generate("Hello, this is a test.", out_path="out.wav")
"""
import sys
import os

import torch
import torchaudio

from .tokenizer import PhonemeTokenizer, AUDIO_VOCAB_SIZE
from .model import DecoderOnlyTTS


class TTSPipeline:
    def __init__(
        self,
        tts_weights: str,
        wavtokenizer_weights: str,
        wavtokenizer_config: str,
        wavtokenizer_repo_path: str = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The vendored WavTokenizer code lives in wavtokenizer/ at the repo
        # root by default; add it to sys.path so `decoder.pretrained` etc.
        # resolve regardless of the caller's cwd.
        if wavtokenizer_repo_path is None:
            wavtokenizer_repo_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "wavtokenizer",
            )
        if wavtokenizer_repo_path not in sys.path:
            sys.path.insert(0, wavtokenizer_repo_path)

        from decoder.pretrained import WavTokenizer  # noqa: E402 (path-dependent import)

        self.text_tokenizer = PhonemeTokenizer()

        self.audio_tokenizer = WavTokenizer.from_pretrained0802(
            wavtokenizer_config, wavtokenizer_weights
        ).to(self.device)
        self.audio_tokenizer.eval()
        self.bandwidth_id = torch.tensor([0]).to(self.device)

        total_vocab_size = AUDIO_VOCAB_SIZE + self.text_tokenizer.vocab_size
        self.model = DecoderOnlyTTS(
            total_vocab_size, pad_idx=self.text_tokenizer.PAD
        ).to(self.device)

        ckpt = torch.load(tts_weights, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        text: str,
        out_path: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        text_ids = torch.tensor(
            self.text_tokenizer.encode(text), dtype=torch.long
        ).unsqueeze(0).to(self.device)

        raw_codes = self.model.generate(
            text_ids,
            sep_id=self.text_tokenizer.SEP,
            eos_id=self.text_tokenizer.EOS,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        clean_codes = [c for c in raw_codes if c < AUDIO_VOCAB_SIZE]
        removed = len(raw_codes) - len(clean_codes)
        print(f"Generated : {len(raw_codes)} tokens | removed {removed} text-range tokens")

        if not clean_codes:
            print("No audio tokens generated — model may need more training.")
            return None

        codes = torch.tensor(clean_codes, dtype=torch.long).unsqueeze(0).to(self.device)
        features = self.audio_tokenizer.codes_to_features(codes)
        audio_out = self.audio_tokenizer.decode(features, bandwidth_id=self.bandwidth_id)
        audio_out = audio_out.squeeze().cpu()

        if out_path:
            torchaudio.save(out_path, audio_out.unsqueeze(0), 24000)
            print(f"Saved to : {out_path}")

        return audio_out
