"""Is this text really Luganda?

Every corpus built from "stuff from Google" is mostly not Luganda. Filtering badly
is the single most expensive mistake in a low-resource project: English text with a
Luganda title teaches the model to answer in English.

No model needed. Luganda has a very distinctive function-word set and heavy
agglutination, so a marker-based scorer beats a naive character check, and it runs
on a laptop. Swap in a trained classifier later if you want (see `classify_trained`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# High-frequency Luganda function words and roots. Deliberately includes the forms
# that actually appear in newspapers and textbooks, not just greetings.
LUG_MARKERS = {
    "nga", "nti", "era", "kubanga", "naye", "ate", "oba", "bwe", "mbu", "anti",
    "mu", "ku", "wa", "ki", "ani", "ddi", "kiki", "lwaki", "ntya", "otya", "kiiki",
    "bulungi", "nnyo", "weebale", "webale", "kale", "ddala", "kati", "leero", "enkya",
    "oli", "ndi", "ali", "tuli", "muli", "bali", "nze", "ggwe", "ye", "ffe", "mmwe",
    "abo", "oyo", "ono", "kino", "kyo", "kye", "eyo", "kuno", "luno", "wano", "awo",
    "aba", "omu", "oku", "eki", "ebi", "ama", "emi", "obu", "olu", "aka", "obu",
    "omuntu", "abantu", "ekintu", "ebintu", "omwana", "abaana", "omukazi", "omusajja",
    "ssente", "emmere", "amazzi", "ennyumba", "essomero", "eddwaaliro", "ekibuga",
    "oluganda", "abaganda", "obuganda", "obuyambi",
    "okukola", "okugenda", "okwogera", "okumanya", "okulaba", "okugula", "okutunda",
    "asobola", "basobola", "tusobola", "nsobola", "kyokka", "nno", "ssebo", "nnyabo",
    "mukwano", "bwino", "mangu", "katono", "kumpi", "wamu", "buli", "bonna", "onna",
    "kino", "ekyo", "ogwo", "agwo", "kiro", "obudde", "emisana", "enkuba", "ejjuba",
    "ssemazinga", "amakuru", "ebiragiro", "obuvunaanyizibwa", "enteekateeka",
}

ENG_MARKERS = {
    "the", "and", "of", "to", "in", "is", "are", "was", "were", "for", "with", "that",
    "this", "from", "have", "has", "had", "will", "would", "should", "not", "you", "we",
    "they", "it", "on", "at", "by", "as", "but", "or", "be", "been", "which", "their",
    "there", "about", "into", "more", "than", "also", "can", "could", "these", "those",
    "government", "people", "said", "according", "police", "news", "report", "minister",
    "million", "billion", "percent", "company", "market", "health", "school", "children",
}

# Swahili contaminates any Ugandan web corpus and shares some morphology with
# Luganda, so it gets its own detector rather than being lumped with "unknown".
SW_MARKERS = {
    "ya", "wa", "kwa", "katika", "na", "ni", "za", "la", "hii", "hilo", "kama",
    "lakini", "pia", "sana", "zaidi", "serikali", "watu", "watoto", "habari",
    "sasa", "kila", "moja", "mbili", "tatu", "nne", "tano", "kutoka", "hadi",
    "kwamba", "alisema", "walisema", "imekuwa", "itakuwa", "wanafunzi", "shule",
    "barabara", "fedha", "waziri", "rais", "nchi", "mkubwa", "ndogo", "vizuri",
}

WORD_RE = re.compile(r"[A-Za-z']+")
# Luganda morphology cue: the noun-class and subject prefixes. English has doubled
# letters ("committee", "all") so geminates alone are too noisy to score on, but
# these prefixes essentially do not occur word-initially in English.
PREFIX_RE = re.compile(
    r"^(aba|abe|eki|ebi|ama|emi|obu|olu|oku|omu|aka|eky|eby|olw|obw|en|em|om|ab|ol|ek|ss|nn|mm|bb|dd|gg|kk|jj|tt|pp)"
)


@dataclass
class LangResult:
    decision: str          # "lug" | "eng" | "mixed" | "unknown"
    lug_score: float
    eng_score: float
    tokens: int
    code_mix: bool

    def as_dict(self) -> dict:
        return asdict(self)


def classify(text: str, *, min_tokens: int = 8) -> LangResult:
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    n = len(tokens)
    if n < min_tokens:
        return LangResult("unknown", 0.0, 0.0, n, False)

    unique = set(tokens)
    marker_hits = sum(1 for t in tokens if t in LUG_MARKERS)
    swahili_hits = sum(1 for t in tokens if t in SW_MARKERS)
    # Morphology alone is a weak signal (Swahili "kutoka", "barabara" both match Luganda
    # prefixes), so it is weighted down: real Luganda always carries marker words too.
    morph_hits = sum(1 for t in unique if PREFIX_RE.search(t))
    lug_score = (0.75 * marker_hits + 0.25 * morph_hits) / n
    eng_score = sum(1 for t in tokens if t in ENG_MARKERS) / n

    sw_score = swahili_hits / n
    if sw_score >= 0.08 and marker_hits < 2:
        decision = "other"          # Swahili (or another Bantu language) — not Luganda
    elif marker_hits >= 2 and lug_score >= 0.05 and lug_score >= eng_score * 1.4:
        decision = "lug"
    elif lug_score >= 0.05 and eng_score >= 0.08:
        decision = "mixed"          # Kampala code-switching — keep, but label it
    elif eng_score >= 0.08:
        decision = "eng"
    elif lug_score > 0.02 or eng_score > 0.02:
        decision = "unknown"        # usually Swahili, Runyankole or a stub
    else:
        decision = "unknown"
    # Code-mixing is normal in Kampala speech (and all over TikTok). Keep it, but
    # label it: it is fine for style/register data and bad for orthography targets.
    code_mix = decision == "mixed" or (decision == "lug" and eng_score > 0.06)
    return LangResult(decision, round(lug_score, 3), round(eng_score, 3), n, code_mix)


TRAINED_HELP = """Optional upgrade: a trained language classifier.
  Sunbird/t5_small_language_Classification on the Hugging Face Hub covers Ugandan
  languages, and fastText's lid.176 covers 'lug' but confuses it with closely related
  languages. Use one only to re-score the 'mixed'/'unknown' bucket — the marker scorer
  above already catches the catastrophic cases (English pages, Swahili, Runyankole).
"""


def classify_trained(texts: list[str], model: str = "Sunbird/t5_small_language_Classification") -> list[str]:
    """Hook for a real classifier. Raises with instructions rather than guessing."""
    try:
        import torch  # noqa: F401, PLC0415
        from transformers import pipeline  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("install torch + transformers for the trained classifier\n" + TRAINED_HELP) from exc
    clf = pipeline("text-classification", model=model, top_k=1)
    return [clf(t[:1000])[0]["label"] for t in texts]


if __name__ == "__main__":  # tiny self-test / playground
    samples = {
        "luganda news": "Pulezidenti yasisinkanye abakulu b'ebitongole era nti gavumenti ejja kuddamu "
                        "okuzimba oluguudo luno mu myezi musanvu.",
        "english news": "The government announced that it will rebuild the road within seven months, "
                        "according to a statement released by the ministry on Tuesday.",
        "code-mixed": "Bino byonna boss, the deadline yaffe is Friday kubanga clients baagala delivery "
                      "before the end of this month.",
        "swahili": "Serikali imetangaza kwamba itajenga upya barabara hii ndani ya miezi saba.",
        "runyankole": "Abaana bagyenda aha ssomero kandi bakagamba ngu barikwija kutwarira "
                      "ebikwata ku byobuhingi omu kyangwe.",
    }
    for name, text in samples.items():
        r = classify(text)
        print(f"{name:14} {r.decision:8} lug={r.lug_score:.2f} eng={r.eng_score:.2f} code_mix={r.code_mix}")
