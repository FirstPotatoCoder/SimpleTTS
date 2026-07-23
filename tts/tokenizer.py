"""
Text-side tokenizer for the TTS model.

Turns raw text -> IPA phonemes (via espeak/phonemizer) -> integer token ids
in the "text" range of the shared audio/text vocabulary (see model.py for
the full id layout).

Requires the `espeak-ng` system package to be installed (see setup.sh).
"""
import re
from phonemizer import phonemize


def clean_text(text: str) -> str:
    text = re.sub(r'\*[^*]+\*', '', text)
    text = text.replace('\u2014', '-')   # em-dash -> hyphen
    text = text.replace('\u2013', '-')   # en-dash -> hyphen too while we're at it
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # curly quotes -> straight
    text = text.replace('\u2026', '.')
    text = text.replace(';', ',')
    text = text.replace(':', ',')
    text = text.replace('\u0329', '')
    text = re.sub(r'["\u201c\u201d\u0329\u00a0]', '', text)
    text = re.sub(r'[^\x00-\x7F\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


IPA_VOCAB = [
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "tʃ", "dʒ",
    "iː", "uː", "ɔː", "ɑː", "ɜː", "eː", "oː",
    "ˈ", "ˌ", "ː",
    "ɪ", "ʊ", "ɛ", "æ", "ʌ", "ɒ", "ə", "ɑ", "ᵻ",
    "ɔ", "ʊ̃", "õ",
    "ɚ", "ɐ", "ɡ", "ɾ",
    "b", "d", "f", "g", "h", "j", "k", "l",
    "m", "n", "p", "r", "s", "t", "v", "w",
    "z", "ð", "θ", "ʃ", "ʒ", "ŋ", "ʔ",
    "ɹ", "ʔ", "x",
    "a", "e", "i", "o", "u",
    " ", ",", ".", "!", "?", " ", '"',
    "ɬ", "̃", "(", ")", "[", "]", "{", "}",

    "<r1>", "<r2>", "<r3>", "<r4>", "<r5>", "<r6>", "<r7>",
    "<r8>", "<r9>", "<r10>", "<r11>", "<r12>",
]

# Audio tokens occupy ids [0, AUDIO_VOCAB_SIZE). Text/special tokens are
# offset above that range so the two vocabularies never collide.
AUDIO_VOCAB_SIZE = 4096


class PhonemeTokenizer:
    def __init__(self):
        OFFSET = AUDIO_VOCAB_SIZE
        SYM_OFFSET = 5

        # Special tokens, offset above the audio vocab.
        self._PAD, self._BOS, self._EOS, self._UNK, self._SEP = [
            idx + OFFSET for idx in range(SYM_OFFSET)
        ]

        # Phoneme vocabulary, offset above the special tokens.
        self._sym2local = {sym: i + SYM_OFFSET + OFFSET for i, sym in enumerate(IPA_VOCAB)}
        self._local2sym = {v: k for k, v in self._sym2local.items()}

        self.vocab_size = len(IPA_VOCAB) + SYM_OFFSET

        # Sort by length descending so multi-char symbols (e.g. "tʃ") are
        # greedily matched before their single-char prefixes.
        sorted_vocab = sorted(IPA_VOCAB, key=len, reverse=True)
        self.split_pattern = re.compile(
            "|".join(re.escape(sym) for sym in sorted_vocab)
        )

    @property
    def PAD(self): return self._PAD

    @property
    def BOS(self): return self._BOS

    @property
    def EOS(self): return self._EOS

    @property
    def UNK(self): return self._UNK

    @property
    def SEP(self): return self._SEP

    def _get_phonemes(self, text: str) -> str:
        cleaned = clean_text(text)
        words = cleaned.split(' ')
        phonemized_words = phonemize(
            words,
            backend="espeak",
            language="en-us",
            with_stress=True,
            preserve_punctuation=True,
            strip=True,
        )
        return ' '.join(phonemized_words)

    def _split(self, phoneme_str: str) -> list[str]:
        tokens, position = [], 0
        while position < len(phoneme_str):
            match = self.split_pattern.match(phoneme_str, position)
            if match:
                tokens.append(match.group(0))
                position = match.end()
            else:
                tokens.append(phoneme_str[position])
                position += 1
        return tokens

    def _ids_from_phoneme_str(self, phoneme_str: str) -> list[int]:
        ids = [self.BOS]
        for t in self._split(phoneme_str):
            ids.append(self._sym2local.get(t, self.UNK))
        ids.append(self.EOS)
        return ids

    def encode(self, text: str) -> list[int]:
        return self._ids_from_phoneme_str(self._get_phonemes(text))

    def batch_encode(self, texts: list[str], njobs: int = 1) -> list[list[int]]:
        cleaned_texts = [clean_text(t) for t in texts]
        word_lists = [t.split(' ') for t in cleaned_texts]
        flat_words = [w for words in word_lists for w in words]

        flat_phonemized = phonemize(
            flat_words,
            backend="espeak",
            language="en-us",
            with_stress=True,
            preserve_punctuation=True,
            strip=True,
            njobs=njobs,
        )

        result = []
        idx = 0
        for words in word_lists:
            n = len(words)
            phoneme_str = ' '.join(flat_phonemized[idx:idx + n])
            result.append(self._ids_from_phoneme_str(phoneme_str))
            idx += n
        return result

    def tokenize(self, text: str) -> list[str]:
        return self._split(self._get_phonemes(text))

    def decode(self, ids: list[int]) -> str:
        result = []
        for i in ids:
            if i in (self._PAD, self._UNK):
                result.append('?')
            elif i in self._local2sym:
                result.append(self._local2sym[i])
        return ''.join(result)
