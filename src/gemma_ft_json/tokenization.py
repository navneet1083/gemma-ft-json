"""Tokenization with a hard no-download guarantee.

Two implementations behind one tiny interface (`BaseTokenizer`):

* `OfflineHFTokenizer` — wraps a Gemma tokenizer loaded FROM A LOCAL DIRECTORY in
  offline mode. Used when you have real Gemma assets on disk.

* `ByteTokenizer` — a pure-python UTF-8 *byte-level* tokenizer needing NO files
  and NO downloads. Byte-level is loss-less for arbitrary JSON text and never
  produces OOV tokens; it lets the ENTIRE pipeline run before Gemma weights exist.

Both expose encode/decode, pad/bos/eos ids, vocab_size, and `added_token_ids`
(the reserved structure/loc tokens), so downstream code is backend-agnostic.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from .exceptions import WeightsNotFoundError

_STRUCTURE_TOKENS = ["<table>", "</table>", "<row>", "</row>",
                     "<cell>", "</cell>", "<eor>", "<eot>"]


def _loc_tokens(n_bins: int) -> List[str]:
    """Quantized coordinate tokens <loc0000>... for inline bbox grounding."""
    width = max(4, len(str(max(0, n_bins - 1))))
    return [f"<loc{str(i).zfill(width)}>" for i in range(max(0, n_bins))]


@runtime_checkable
class BaseTokenizer(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int
    vocab_size: int

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 byte-level tokenizer. No external assets required."""

    def __init__(self, n_structure_tokens: int = 8, n_loc_bins: int = 0):
        self.pad_id, self.bos_id, self.eos_id = 256, 257, 258
        next_id = 259
        self.added_tokens: Dict[str, int] = {}
        for tok in _STRUCTURE_TOKENS[: max(0, n_structure_tokens)]:
            self.added_tokens[tok] = next_id; next_id += 1
        for tok in _loc_tokens(n_loc_bins):
            self.added_tokens[tok] = next_id; next_id += 1
        self.vocab_size = next_id
        self._id_to_added = {v: k for k, v in self.added_tokens.items()}

    @property
    def added_token_ids(self) -> List[int]:
        return list(self.added_tokens.values())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [self.bos_id] if add_bos else []
        ids.extend(text.encode("utf-8"))  # raw bytes; HF path handles subword merges
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if i in self._id_to_added:
                out.extend(self._id_to_added[i].encode("utf-8"))
            elif 0 <= i < 256:
                out.append(i)
        return out.decode("utf-8", errors="replace")


class OfflineHFTokenizer:
    """Wraps a Gemma tokenizer loaded strictly from a local directory, offline."""

    def __init__(self, local_dir: str, n_structure_tokens: int = 8, n_loc_bins: int = 0):
        try:
            from transformers import AutoTokenizer
        except Exception as e:  # noqa: BLE001
            raise WeightsNotFoundError(
                "transformers required for the offline HF tokenizer "
                "(`pip install -e '.[hf]'`). Original: " + str(e)
            ) from e
        import os
        if not local_dir or not os.path.isdir(local_dir):
            raise WeightsNotFoundError(
                f"Local Gemma tokenizer dir not found: '{local_dir}'. No download attempted."
            )
        # local_files_only=True + global offline env vars => any network attempt raises.
        self._tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        self._added_strs = _STRUCTURE_TOKENS[: max(0, n_structure_tokens)] + _loc_tokens(n_loc_bins)
        self._tok.add_tokens(self._added_strs)
        self.pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id
        self.bos_id = self._tok.bos_token_id if self._tok.bos_token_id is not None else self.pad_id
        self.eos_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else self.pad_id
        self.vocab_size = len(self._tok)

    @property
    def added_token_ids(self) -> List[int]:
        return [self._tok.convert_tokens_to_ids(t) for t in self._added_strs]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    @property
    def hf(self):
        """Underlying HF tokenizer (e.g. to resize model embeddings)."""
        return self._tok


def build_tokenizer(decoder_cfg) -> BaseTokenizer:
    """Factory: tokenizer matching the decoder backend."""
    if decoder_cfg.backend == "local_gemma":
        return OfflineHFTokenizer(
            decoder_cfg.local_dir,
            n_structure_tokens=decoder_cfg.n_structure_tokens,
            n_loc_bins=decoder_cfg.n_loc_bins,
        )
    return ByteTokenizer(
        n_structure_tokens=decoder_cfg.n_structure_tokens,
        n_loc_bins=decoder_cfg.n_loc_bins,
    )
