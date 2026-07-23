"""
Decoder-only transformer TTS model.

Token id layout (no overlap):
    0 .. 4095  : audio tokens  (WavTokenizer codebook)
    4096       : PAD  (padding_idx — zero gradient, safe)
    4097       : BOS
    4098       : EOS  (shared text / audio)
    4099       : UNK
    4100       : SEP
    4101+      : phoneme symbols

These hyperparameters must exactly match whatever the checkpoint was
trained with, or load_state_dict() will fail / silently mismatch shapes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import AUDIO_VOCAB_SIZE

TEXT_OFFSET = AUDIO_VOCAB_SIZE

# ── Hyperparameters ──────────────────────────────────────────────────────
BLOCK_SIZE = 1536
N_EMBD     = 576
N_HEAD     = 8
N_LAYER    = 12
DROPOUT    = 0.1


class CausalSelfAttention(nn.Module):
    """
    - Single fused QKV projection instead of n_head separate Linear layers
      (fewer, bigger matmuls = better GPU utilization).
    - Supports an optional KV cache for O(1)-per-step generation instead of
      O(T) recompute every step.
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head    = n_head
        self.head_size = n_embd // n_head
        self.n_embd    = n_embd

        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        """
        x: (B, T, C)  -- T is the number of NEW tokens this call
           (full sequence length during training, 1 (or a few) during
           incremental generation)
        attention_mask: (B, T_total) over the *real* tokens including any past
           cached tokens + the new ones (1 = real, 0 = pad). Pass None during
           generation since there's no padding there.
        past_kv: optional tuple (past_k, past_v), each (B, n_head, T_past, head_size)
        use_cache: if True, returns the updated (k, v) cache alongside output
        """
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, n_head, T_past+T, head_size)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        T_q = q.shape[2]
        T_k = k.shape[2]

        if T_q == T_k:
            causal = self.tril[:T_q, :T_k]
        else:
            # Incremental decode: new queries (rows) may attend to ALL past
            # keys plus themselves causally.
            past_len = T_k - T_q
            row = torch.arange(T_q, device=x.device).unsqueeze(1)
            col = torch.arange(T_k, device=x.device).unsqueeze(0)
            causal = (col <= row + past_len)

        attn_mask = causal.view(1, 1, T_q, T_k)
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :].bool()  # (B,1,1,T_k)
            attn_mask = attn_mask & pad

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )  # (B, n_head, T_q, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        out = self.resid_dropout(self.proj(out))
        return out, new_kv


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa   = CausalSelfAttention(n_embd, n_head, BLOCK_SIZE, DROPOUT)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        attn_out, new_kv = self.sa(self.ln1(x), attention_mask, past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class DecoderOnlyTTS(nn.Module):
    def __init__(self, total_vocab_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(total_vocab_size, N_EMBD, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.ModuleList([Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, total_vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None,
                past_key_values=None, use_cache=False, position_ids=None):
        """
        input_ids: (B, T) - full sequence during training, or just the NEW
                   token(s) during incremental generation with a cache.
        attention_mask: (B, T_total) over past+new real tokens (1=real,0=pad).
                   Pass None when there's no padding (e.g. plain generation).
        past_key_values: list of (k, v) tuples, one per layer, or None.
        position_ids: (B, T) explicit positions for the NEW tokens. If None,
                   computed assuming input_ids start at position 0 (training
                   default) or continue from the cache length (generation).
        """
        B, T = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        tok_emb = self.embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = tok_emb + pos_emb

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, attention_mask, layer_past, use_cache)
            if use_cache:
                new_caches.append(new_kv)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            B_, T_, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B_ * T_, V),
                labels.reshape(B_ * T_),
                ignore_index=-100,
            )

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, text_ids, sep_id, eos_id, max_new_tokens=512,
                 temperature=1.0, top_k=50):
        """
        text_ids: (1, S) global text token ids. Returns list of audio ids.
        Uses a KV cache: the prompt is processed once, then each subsequent
        step only computes the new token instead of reprocessing everything.

        sep_id / eos_id: pass text_tokenizer.SEP / text_tokenizer.EOS from
        the caller (kept as explicit args rather than class-level defaults
        so this module has no import-time dependency on a tokenizer instance).
        """
        self.eval()
        device = text_ids.device
        input_ids = torch.cat([
            text_ids,
            torch.tensor([[sep_id]], device=device),
        ], dim=1)

        # ── Prefill: one forward pass over the whole prompt, builds the cache.
        logits, _, past_key_values = self.forward(input_ids, use_cache=True)
        logits = logits[:, -1, :]

        generated = []
        cur_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            if cur_len >= BLOCK_SIZE:
                break

            scaled = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(scaled, min(top_k, scaled.shape[-1]))
                scaled[scaled < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1,1)

            if next_tok.item() == eos_id:
                break

            generated.append(next_tok.item())
            cur_len += 1

            logits, _, past_key_values = self.forward(
                next_tok, past_key_values=past_key_values, use_cache=True,
            )
            logits = logits[:, -1, :]

        return generated
