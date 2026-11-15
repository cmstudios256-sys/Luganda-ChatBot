"""Automatic metrics for Luganda generation: chrF++ and corpus BLEU-4.

Extracted so tests can import them without pulling in the app or argparse.
"""
from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f_score(matches: int, total_hyp: int, total_ref: int, beta: float = 2.0) -> float:
    if total_hyp == 0 or total_ref == 0 or matches == 0:
        return 0.0
    p = matches / total_hyp
    r = matches / total_ref
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (p or r) else 0.0


def chrf_plus_plus(hypothesis: str, reference: str, char_order: int = 6, word_order: int = 2, beta: float = 2.0) -> float:
    """chrF++ (Popović 2017): character n-grams 1..6 + word n-grams 1..2, beta=2.

    Corpus-consistent approximation of sacrebleu's implementation: character
    n-grams are long so individual sentences score low; compare runs, not papers.
    """
    hyp, ref = hypothesis.strip(), reference.strip()
    if not hyp or not ref:
        return 0.0
    matches = hyp_total = ref_total = 0
    for n in range(1, char_order + 1):
        h, r = _ngrams(list(hyp), n), _ngrams(list(ref), n)
        matches += sum((h & r).values())
        hyp_total += sum(h.values())
        ref_total += sum(r.values())
    for n in range(1, word_order + 1):
        h, r = _ngrams(hyp.split(), n), _ngrams(ref.split(), n)
        matches += sum((h & r).values())
        hyp_total += sum(h.values())
        ref_total += sum(r.values())
    return _f_score(matches, hyp_total, ref_total, beta) * 100


def bleu(corpus: list[tuple[str, str]], max_n: int = 4) -> dict:
    """Corpus BLEU-4 with Lin & Och smoothing (small corpora do not get zeros)."""
    clipped = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in corpus:
        h_tok, r_tok = hyp.split(), ref.split()
        hyp_len += len(h_tok)
        ref_len += len(r_tok)
        for n in range(1, max_n + 1):
            h, r = _ngrams(h_tok, n), _ngrams(r_tok, n)
            clipped[n - 1] += sum(min(c, r[g]) for g, c in h.items())
    precisions = []
    for n in range(1, max_n + 1):
        denom = max(hyp_len - n + 1, 0)
        num = clipped[n - 1]
        if n > 1:                     # add-1 smoothing for higher orders
            num += 1
            denom += 1
        precisions.append(num / denom if denom > 0 else 0.0)
    score = 0.0
    if all(precisions):
        score = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return {"bleu": round(100 * score * bp, 2), "brevity_penalty": round(bp, 3),
            "precisions": [round(p, 3) for p in precisions]}
