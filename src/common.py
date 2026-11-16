"""Shared constants and helpers for the dataset pipeline (Week 3).

Frege format:  [FUNCTION_START] <instruction> [FUNCTION_END] [ARGUMENT_START] <data> [ARGUMENT_END]
function = the trusted instruction, argument = the untrusted data it is applied to.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW, INTERIM, PROCESSED = ROOT / "data/raw", ROOT / "data/interim", ROOT / "data/processed"
REPORTS, FIGURES = ROOT / "reports", ROOT / "figures"
for _p in (INTERIM, PROCESSED, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --- Frege format ------------------------------------------------------------
FS, FE = "[FUNCTION_START]", "[FUNCTION_END]"
AS, AE = "[ARGUMENT_START]", "[ARGUMENT_END]"
_MARKER_RE = re.compile(r"\[\s*(FUNCTION|ARGUMENT)\s*_\s*(START|END)\s*\]", re.IGNORECASE)


def escape_markers(text: str) -> str:
    """Neutralise reserved markers inside untrusted text: '[X_Y]' becomes '(X_Y)'.

    Replacement (not deletion) cannot create a new marker by concatenation, e.g.
    '[FUNCTION_[FUNCTION_START]START]' stays harmless. The loop is only a safety net.
    LIMITATION: this is string-level escaping. Real isolation needs reserved tokens that
    the tokenizer never produces from user text (the StruQ approach) - note this in the thesis.
    """
    prev = None
    while prev != text:
        prev = text
        text = _MARKER_RE.sub(lambda m: f"({m.group(1).upper()}_{m.group(2).upper()})", text)
    return text


def wrap(instruction: str, data: str) -> str:
    return f"{FS} {escape_markers(instruction)} {FE} {AS} {escape_markers(data)} {AE}"


def is_well_formed(text: str) -> bool:
    """Exactly one of each marker, in the right order."""
    if any(text.count(m) != 1 for m in (FS, FE, AS, AE)):
        return False
    return text.index(FS) < text.index(FE) < text.index(AS) < text.index(AE)


# --- dataset schema ----------------------------------------------------------
COLUMNS = ["doc_id", "variant", "instruction", "data", "label", "attack_family",
           "hn_type", "position", "payload", "task", "source",
           "template_test_only", "hn_test_only"]
POSITIONS = ["start", "middle", "end"]
# Same style prefixes are used for attacks AND harmless notes, so a prefix is not a shortcut.
STYLE_PREFIXES = ["", "", "Note: ", "Important: ", "P.S. ", "Reminder: ", "Update: "]

TASKS = [
    ("summarize", "Summarize the following passage in two sentences."),
    ("title", "Give the following passage a short, descriptive title."),
    ("facts", "List the three most important facts in the following passage."),
    ("extract", "Extract the names, dates and places mentioned in the following text."),
    ("simplify", "Rewrite the following passage in simpler language."),
    ("qa", "Answer the question using only the following passage: {question}"),
]


def make_instruction(task: str, question: str) -> str:
    template = dict(TASKS)[task]
    return template.format(question=question) if "{question}" in template else template


# --- text helpers --------------------------------------------------------------
def doc_id_of(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


_EN_STOP = set("the a an of to and in is that it for with as on be this are was you your i not or at by "
               "from have has but they their which will can".split())


def looks_english(text: str, min_ratio: float = 0.2, min_tokens: int = 4) -> bool:
    """Cheap heuristic (share of common English words) used to drop German/other payloads,
    so that language does not become a shortcut. Not a real language detector."""
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if len(toks) < min_tokens:
        return False
    return sum(t in _EN_STOP for t in toks) / len(toks) >= min_ratio


def split_sentences(text: str):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def insert_snippet(passage: str, snippet: str, position: str, rng) -> str:
    sents = split_sentences(passage)
    if position == "start":
        i = 0
    elif position == "end":
        i = len(sents)
    else:  # middle
        i = int(rng.integers(1, len(sents))) if len(sents) >= 2 else len(sents)
    sents.insert(i, snippet)
    return " ".join(sents)


# --- domain pool & URL helpers -------------------------------------------------
FAKE_DOMAINS = [
    "attacker.example", "evil.test", "hacker.invalid",
    "steal-data.example", "not-suspicious.test", "legit-login.example",
    "secure-verify.test", "account-check.invalid", "update-now.example",
    "verify-identity.test",
]

BENIGN_URL_PATHS = [
    "report", "overview", "docs", "data", "summary",
    "archive", "resources", "article", "figure", "charts",
]


def make_benign_url(rng) -> str:
    """Generate a harmless URL matching the domain pool and path style used by attack payloads."""
    proto = str(rng.choice(["https", "http"]))
    domain = str(rng.choice(FAKE_DOMAINS))
    path = str(rng.choice(BENIGN_URL_PATHS))
    return f"{proto}://{domain}/{path}"


# --- loaders -------------------------------------------------------------------
def load_squad_docs() -> pd.DataFrame:
    """One row per unique squad_v2 passage (first question kept) with a stable doc_id."""
    df = pd.read_parquet(RAW / "squad_v2_train.parquet", columns=["context", "question"])
    df = df.dropna().drop_duplicates("context").reset_index(drop=True)
    df["doc_id"] = df["context"].map(doc_id_of)
    return df


def load_deepset_attacks(max_chars: int = 300) -> list:
    """Real injection prompts (label 1), English only, short enough to be inserted into a passage."""
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("prompt_injections_*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    texts = df.loc[df.label == 1, "text"].tolist()
    return [t for t in texts if 20 <= len(t) <= max_chars and looks_english(t)]
