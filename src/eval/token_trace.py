"""Utilities for token-trace analysis from saved TTS candidate outputs."""

from __future__ import annotations

from collections import Counter


def _clean_token_piece(tok: str) -> str:
    """Normalize tokenizer-specific token text for display and matching."""
    return str(tok).replace("Ġ", "").strip()


def token_lengths(candidate_tokens: list[list[str]]) -> list[int]:
    """Return token count per candidate."""
    return [len(row) for row in candidate_tokens]


def token_position_agreement(candidate_tokens: list[list[str]]) -> list[float]:
    """Agreement rate at each token position across candidates.

    Agreement at position p is:
    max_token_count_at_p / number_of_candidates_with_a_token_at_p.
    """
    if not candidate_tokens:
        return []

    max_len = max((len(row) for row in candidate_tokens), default=0)
    out: list[float] = []

    for pos in range(max_len):
        toks = [_clean_token_piece(row[pos]) for row in candidate_tokens if pos < len(row)]
        toks = [t for t in toks if t]
        if not toks:
            out.append(0.0)
            continue
        counts = Counter(toks)
        top = max(counts.values())
        out.append(top / len(toks))

    return out


def token_majority_sequence(candidate_tokens: list[list[str]]) -> list[str]:
    """Build majority token sequence over positions with first-seen tie break."""
    if not candidate_tokens:
        return []

    max_len = max((len(row) for row in candidate_tokens), default=0)
    out: list[str] = []

    for pos in range(max_len):
        toks = [_clean_token_piece(row[pos]) for row in candidate_tokens if pos < len(row)]
        toks = [t for t in toks if t]
        if not toks:
            break
        counts = Counter(toks)
        top = max(counts.values())
        tied = {t for t, c in counts.items() if c == top}
        chosen = next(t for t in toks if t in tied)
        out.append(chosen)

    return out
