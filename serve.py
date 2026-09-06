"""
Web server for the Lun Bawang ↔ English translator.

Usage:
  python3.13 serve.py              # runs on http://localhost:8000
  python3.13 serve.py --port 8080  # custom port

The backend reads tinker_state.json to find the latest checkpoint and
calls the Tinker OpenAI-compatible serving API for inference.
"""

import base64
import csv
import io
import json
import os
import re
import argparse
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────

STATE_FILE         = Path(__file__).parent / "tinker_state.json"
STATE_FILE_V1      = Path(__file__).parent / "tinker_state_v1.json"
STATE_FILE_RUN1    = Path(__file__).parent / "tinker_state_run1.json"
STATE_FILE_INKLING = Path(__file__).parent / "tinker_state_inkling.json"
INKLING_CHECKPOINT_EVERY = 2000  # only expose every Nth Inkling checkpoint in the dropdown
# The v2 (dictionary) run — warm-started from Inkling checkpoint-11500 and still
# training. Its checkpoints are exposed in the compare dropdown for evaluation
# but are NOT the serving default (see get_latest_checkpoint, which is unchanged).
STATE_FILE_INKLING_V2 = Path(__file__).parent / "tinker_state_inkling_v2.json"
INKLING_V2_CHECKPOINT_EVERY = 2000  # only expose every Nth v2 checkpoint in the dropdown
DEFAULT_V2_STEP = 16000  # the v2 checkpoint served as the production default (see get_latest_checkpoint)

STATIC_DIR  = Path(__file__).parent / "static"
API_KEY     = os.environ["TINKER_API_KEY"]
TINKER_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Tinker cold-starts a sampler that hasn't been used recently: measured 14–21s
# for the first call vs ~2s once warm, on a checkpoint idle for a couple of
# hours. The frontend pings /api/warmup as soon as someone shows intent to
# type, so the load overlaps with them composing their query instead of
# landing on it. This throttle caps the cost: however much traffic arrives,
# a given checkpoint is only actually warmed once per interval.
WARMUP_MIN_INTERVAL = 60.0        # seconds
_last_warmup: dict[str, float] = {}
_warmup_lock = threading.Lock()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO  = "matteuspan/lunbawang-translate"
GITHUB_PATH  = "eval/feedback.csv"

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate ONLY the exact text provided — output just the translation, nothing else. "
    "Use everyday conversational language, not religious or scriptural register. "
    "Proper names (e.g. Bethel, Joyce, Sarah) are names of ordinary people — do not treat them as biblical references. "
    "Preserve proper nouns (personal names, place names) exactly as they appear in the input unless you are certain of the standard equivalent in the target language. "
    "Do not expand, paraphrase, or add any meaning not present in the input. "
    "Do not produce Bible verse language."
)

# Common English words for language auto-detection.
# Covers function words, time words, common nouns/verbs/adjectives so that
# everyday English content words (e.g. "tomorrow") are reliably detected.
ENGLISH_WORDS = {
    # Function words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "into", "year", "your", "good", "some", "could", "them", "see",
    "than", "then", "now", "look", "only", "come", "its", "over", "how",
    "our", "first", "well", "even", "new", "want", "because", "any",
    "these", "most", "us", "is", "are", "was", "were", "has", "had",
    "been", "am", "god", "lord", "said", "shall", "did", "does",
    "i", "those", "through", "before", "after", "many", "also", "where",
    "much", "must", "upon", "great", "against", "between", "down",
    "why", "while", "though", "although", "however", "therefore",
    "else", "either", "both", "each", "every", "another", "other", "such",
    "few", "enough", "whole", "together", "along", "since", "during",
    "without", "within", "behind", "above", "below", "under", "across",
    # Time
    "today", "tomorrow", "yesterday", "morning", "evening", "night", "day",
    "week", "month", "noon", "midnight", "soon", "always", "never", "often",
    "again", "already", "still", "yet", "once", "twice", "early", "late",
    # Common nouns
    "house", "home", "water", "food", "name", "man", "woman", "child",
    "people", "place", "thing", "way", "hand", "eye", "face", "head",
    "body", "heart", "mind", "life", "world", "country", "land", "earth",
    "sky", "sun", "moon", "star", "fire", "river", "tree", "road", "door",
    "work", "love", "friend", "family", "father", "mother", "son", "daughter",
    "brother", "sister", "king", "blood", "word", "voice", "light",
    "book", "school", "church", "village", "town", "city", "field",
    "dog", "bird", "fish", "horse", "pig", "cat", "cow",
    # Common verbs
    "find", "ask", "feel", "try", "leave", "call", "keep", "let", "begin",
    "show", "hear", "play", "run", "move", "live", "believe", "hold", "bring",
    "write", "sit", "stand", "lose", "meet", "continue", "set", "learn",
    "change", "follow", "stop", "speak", "read", "grow", "open", "walk",
    "remember", "consider", "appear", "buy", "wait", "serve", "die", "send",
    "build", "stay", "fall", "cut", "reach", "remain", "raise", "pass",
    "eat", "drink", "sleep", "pray", "sing", "give", "receive", "return",
    # Common adjectives
    "long", "last", "little", "own", "right", "big", "high", "different",
    "small", "large", "next", "young", "important", "bad", "same", "able",
    "old", "free", "real", "best", "better", "sure", "true", "hard",
    "possible", "strong", "white", "black", "red", "blue", "green",
    "hot", "cold", "beautiful", "happy", "sad", "angry", "tired", "ready",
    "dead", "full", "close", "short", "certain", "low", "clear",
    "holy", "righteous", "faithful", "eternal", "blessed", "mighty",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Returns 'lb2en' (Lun Bawang input) or 'en2lb' (English input)."""
    words = [w.strip(".,!?;:'\"()[]") for w in text.lower().split()]
    words = [w for w in words if w]
    if not words:
        return "lb2en"
    english_count = sum(1 for w in words if w in ENGLISH_WORDS)
    if english_count >= 2 or english_count / len(words) >= 0.25:
        return "en2lb"
    return "lb2en"


# ── Phrasebook (deterministic exact-match lookup) ───────────────────────────
#
# Short conversational phrases are how most people judge a translator, and they
# are also where the model is least reliable (single-token empty outputs,
# lexical failures on greetings). For a small, hand-verified set we bypass the
# model entirely and return a fixed answer. Two properties drive the design:
#
#   * Consistency with the website. Every pair below is the verbatim phrase
#     glossary shown on the homepage (static/index.html), so what the
#     translator returns for "Good morning" always matches what the on-page
#     glossary teaches.
#   * Precision over recall. These answers are served at 100% confidence with
#     no model in the loop, so a wrong entry is worse than a model guess. We
#     only include pairs we can vouch for, and only fire on an exact, whole-
#     input match (case- and trailing-punctuation-insensitive) — never partial
#     or fuzzy. Ambiguous candidates are deliberately left out (e.g. "hello":
#     the feedback snapshot shows two competing forms, "sini'" and "Alo", so it
#     is not safe to serve deterministically).
#
# The lookup runs only for the default production checkpoint (see translate());
# the checkpoint-compare dropdown still shows raw model output so evaluations
# aren't masked by the phrasebook.

# (lun_bawang, english). Direction-agnostic source of truth; both lookup maps
# derive from it. The first 12 pairs are the homepage glossary verbatim.
PHRASEBOOK: list[tuple[str, str]] = [
    ("Do pekak", "Good morning"),
    ("Ro meco", "Good evening"),
    ("Ro lawe", "Goodbye"),
    ("Mo", "Yes"),
    ("Nam", "No"),
    ("Anun bala?", "How are you?"),
    ("Ui mawa nemu", "I love you"),
    ("Tinam", "Mother"),
    ("Tamam", "Father"),
    ("Rurum", "Friend"),
    ("Aceh, dueh, teluh", "One, two, three"),
    ("Kuman", "Eat"),
    # Not on the on-page glossary but a very common test word, and well attested
    # for "water" across three independent corpus sources (borneodictionary,
    # longsemadoh, mortensen: abpa'/abpa/ebpa').
    ("Abpa'", "Water"),
]


def _pb_norm(s: str) -> str:
    """Normalize a phrase for exact-match lookup: fold case, collapse internal
    whitespace, and drop trailing terminal punctuation. Deliberately keeps
    internal punctuation significant (the commas in 'Aceh, dueh, teluh' must be
    typed) — the match is on the whole phrase, never fuzzy or partial."""
    s = re.sub(r"\s+", " ", s.strip()).lower()
    return s.rstrip("?!.,;:")


_PB_EN2LB = {_pb_norm(en): lb for lb, en in PHRASEBOOK}
_PB_LB2EN = {_pb_norm(lb): en for lb, en in PHRASEBOOK}


def phrasebook_lookup(text: str, direction: str):
    """Return (translation, resolved_direction, detected_lang) for an exact
    full-phrase hit, else None.

    An explicit direction ("en2lb"/"lb2en") only consults that side. "auto"
    tries the English side first, then the Lun Bawang side, and the side that
    matches also fixes the direction — so known phrases the heuristic detector
    would mislabel (e.g. "Do pekak", where "do" reads as an English word) are
    still routed correctly. The two maps share no keys, so the order is safe."""
    key = _pb_norm(text)
    if direction == "en2lb":
        lb = _PB_EN2LB.get(key)
        return (lb, "en2lb", "en") if lb else None
    if direction == "lb2en":
        en = _PB_LB2EN.get(key)
        return (en, "lb2en", "lb") if en else None
    # auto — let the matching side decide the direction
    lb = _PB_EN2LB.get(key)
    if lb:
        return (lb, "en2lb", "en")
    en = _PB_LB2EN.get(key)
    if en:
        return (en, "lb2en", "lb")
    return None


def strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>…</think> reasoning blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def clean_translation(text: str, source: str = "") -> str:
    """Strip trailing punctuation that bleeds from Bible verse training data,
    then mirror the source's terminal punctuation (. ! ?) if it had any."""
    text = text.rstrip(".,;:").strip()
    if source:
        last = source.rstrip()[-1] if source.rstrip() else ""
        if last in ".!?":
            text += last
    return text


# Coordinating conjunctions that warrant a clause split when each side >= 3 words
_CONJ_RE = re.compile(r"\s+(?:and|but|or|so|yet|nor)\s+", re.IGNORECASE)

def split_clauses(text: str) -> list[str]:
    """Split English text into clauses at commas, semicolons, and coordinating
    conjunctions (only when each side has >= 3 words to avoid splitting phrases
    like 'bread and butter')."""
    # First split on commas and semicolons
    parts = re.split(r"\s*[,;]\s*", text)

    # Then further split on conjunctions where both sides are substantial
    result = []
    for part in parts:
        sub = _CONJ_RE.split(part)
        if len(sub) > 1 and all(len(s.split()) >= 3 for s in sub):
            result.extend(sub)
        else:
            result.append(part)

    return [p.strip() for p in result if p.strip()]


def get_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _all_checkpoints() -> list[dict]:
    """Return all checkpoints across all runs, newest first."""
    current = get_state().get("checkpoints", [])
    if current:
        return current
    # No active run — fall back to most recent archived run
    if STATE_FILE_V1.exists():
        ckpts = json.loads(STATE_FILE_V1.read_text()).get("checkpoints", [])
        if ckpts:
            return ckpts
    if STATE_FILE_RUN1.exists():
        return json.loads(STATE_FILE_RUN1.read_text()).get("checkpoints", [])
    return []


def get_latest_checkpoint() -> str | None:
    """The v2 dictionary-retrain · checkpoint-16000 is the production default.
    It was warm-started from Inkling checkpoint-11500 and trained on the Kemaloh
    Lundayeh dictionary; on the everyday-quality blend (dict-sentence BLEU both
    directions + general-sentence BLEU + dict-word chrF), measured at the same
    "medium" reasoning effort the server uses, it is the best checkpoint of the
    run — strongest general lb→en sentences and en→lb dictionary quality, with
    Bible fidelity holding (~57 BLEU, no collapse). Pinned by step so it can't
    drift as checkpoints are added. Falls back to Inkling checkpoint-11500, then
    the Qwen v1/v0 chain, if the v2 state file is unavailable."""
    if STATE_FILE_INKLING_V2.exists():
        v2_ckpts = json.loads(STATE_FILE_INKLING_V2.read_text()).get("checkpoints", [])
        for ck in v2_ckpts:
            if ck["step"] == DEFAULT_V2_STEP:
                return ck["path"]
    if STATE_FILE_INKLING.exists():
        inkling_ckpts = json.loads(STATE_FILE_INKLING.read_text()).get("checkpoints", [])
        if inkling_ckpts:
            return inkling_ckpts[-1]["path"]
    ckpts = _all_checkpoints()
    return ckpts[-1]["path"] if ckpts else None


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(title="LunBawang Translate")


class TranslateRequest(BaseModel):
    text: str
    direction: str = "auto"   # "auto" | "lb2en" | "en2lb"
    checkpoint: str | None = None  # None = use latest
    clause_split: bool = False


class FeedbackRequest(BaseModel):
    source_text: str
    direction: str
    checkpoint: str
    model_output: str
    rating: int           # 1 or -1
    correction: str | None = None


class WarmupRequest(BaseModel):
    checkpoint: str | None = None  # None = use the default checkpoint


@app.get("/api/status")
def status():
    state = get_state()
    checkpoint = get_latest_checkpoint()
    return {
        "ready": checkpoint is not None,
        "checkpoint": checkpoint,
        "steps": state.get("steps", 0),
        "num_checkpoints": len(list_checkpoints()),
    }


@app.get("/api/checkpoints")
def list_checkpoints():
    def _load(path):
        return json.loads(path.read_text()) if path.exists() else {}

    run1_state = _load(STATE_FILE_RUN1)
    v1_state   = _load(STATE_FILE_V1)
    main_state = get_state()

    run1_model_id = run1_state.get("model_id", "")
    v1_model_id   = v1_state.get("model_id", "")
    main_model_id = main_state.get("model_id", "")

    run1_ckpts = run1_state.get("checkpoints", [])
    v1_ckpts   = v1_state.get("checkpoints", [])
    main_ckpts = main_state.get("checkpoints", [])

    # If main state file matches an archived run (e.g. on Render where
    # tinker_state.json is the committed copy), merge into that run's list.
    if main_model_id and main_model_id == run1_model_id:
        merged = {ck["step"]: ck for ck in run1_ckpts}
        merged.update({ck["step"]: ck for ck in main_ckpts})
        run1_ckpts = sorted(merged.values(), key=lambda x: x["step"])
        main_ckpts = []
    elif main_model_id and main_model_id == v1_model_id:
        merged = {ck["step"]: ck for ck in v1_ckpts}
        merged.update({ck["step"]: ck for ck in main_ckpts})
        v1_ckpts = sorted(merged.values(), key=lambda x: x["step"])
        main_ckpts = []

    result = []

    for ck in run1_ckpts:
        label = ck.get("label") or f"v0 · Step {ck['step']:,}"
        if "epoch" in ck and not ck.get("label"):
            label += f" · Epoch {ck['epoch']}"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    for ck in v1_ckpts:
        label = ck.get("label") or f"v1 · Step {ck['step']:,}"
        if "epoch" in ck and not ck.get("label"):
            label += f" · Epoch {ck['epoch']}"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    for ck in main_ckpts:
        label = ck.get("label") or f"v2 · Step {ck['step']:,}"
        if "epoch" in ck and not ck.get("label"):
            label += f" · Epoch {ck['epoch']}"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    # Inkling-Small run — now the production default (see README). Only
    # every Nth checkpoint is exposed (the run saves one every 500 steps;
    # showing all of them would flood the dropdown), plus the final
    # checkpoint-11500 explicitly since it may not land on that boundary.
    inkling_state = _load(STATE_FILE_INKLING)
    inkling_all = inkling_state.get("checkpoints", [])
    inkling_ckpts = [ck for ck in inkling_all if ck["step"] % INKLING_CHECKPOINT_EVERY == 0]
    if inkling_all and inkling_all[-1] not in inkling_ckpts:
        inkling_ckpts.append(inkling_all[-1])
    for ck in inkling_ckpts:
        label = f"Inkling · Step {ck['step']:,}"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    # Inkling v2 (dictionary) run — selectable for comparison but never the
    # default. Same every-Nth thinning as the base Inkling run, plus the newest
    # checkpoint so the freshest weights are always reachable.
    inkling_v2_state = _load(STATE_FILE_INKLING_V2)
    inkling_v2_all = inkling_v2_state.get("checkpoints", [])
    inkling_v2_ckpts = [ck for ck in inkling_v2_all if ck["step"] % INKLING_V2_CHECKPOINT_EVERY == 0]
    if inkling_v2_all and inkling_v2_all[-1] not in inkling_v2_ckpts:
        inkling_v2_ckpts.append(inkling_v2_all[-1])
    for ck in inkling_v2_ckpts:
        label = f"Inkling v2 (dict) · Step {ck['step']:,}"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    # Drop checkpoints whose weights were pruned from Tinker to save storage
    # (tagged weights_deleted in the state files — their metadata/metrics are
    # kept for the record, but they can no longer be served, so they must not
    # appear as selectable options). Collected across every state file.
    deleted_paths = set()
    for st in (_load(STATE_FILE_RUN1), _load(STATE_FILE_V1), _load(STATE_FILE),
               _load(STATE_FILE_INKLING), _load(STATE_FILE_INKLING_V2)):
        deleted_paths.update(c["path"] for c in st.get("checkpoints", [])
                             if c.get("weights_deleted"))
    result = [e for e in result if e["path"] not in deleted_paths]

    # Mark whichever entry is the actual serving default (see
    # get_latest_checkpoint) rather than assuming it's the last bucket.
    default_checkpoint = get_latest_checkpoint()
    for entry in result:
        if entry["path"] == default_checkpoint:
            entry["label"] += " (default)"

    return result


def _inkling_checkpoint_paths() -> set[str]:
    """Paths for every Inkling-family checkpoint (base run + v2 dictionary run).
    Both share the thinkingmachines TML base, so both need reasoning_effort set
    on completions — grouping them here keeps warmup and translate() correct for
    v2 checkpoints without any further branching."""
    paths: set[str] = set()
    for state_file in (STATE_FILE_INKLING, STATE_FILE_INKLING_V2):
        if state_file.exists():
            state = json.loads(state_file.read_text())
            paths.update(ck["path"] for ck in state.get("checkpoints", []))
    return paths


def _warm_sampler(checkpoint: str) -> None:
    """Send one tiny throwaway completion so Tinker loads the sampler.

    Best-effort by design: nothing depends on the result, and any failure
    (network, cold-start timeout, bad checkpoint) must not surface to the
    user — they'll simply pay the cold start on their real query, which is
    the status quo this is trying to improve on.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=TINKER_BASE)
        kwargs = {"reasoning_effort": "medium"} if checkpoint in _inkling_checkpoint_paths() else {}
        client.chat.completions.create(
            model=checkpoint,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": "Translate to English:\nmo"}],
            max_tokens=8, temperature=0.1, top_p=0.9, **kwargs,
        )
    except Exception:
        pass


@app.post("/api/warmup")
def warmup(req: WarmupRequest, background_tasks: BackgroundTasks):
    """Fire-and-forget sampler warm-up, called when a user looks like they're
    about to translate. Returns immediately; the actual call runs in the
    background so the client never waits on it."""
    checkpoint = req.checkpoint or get_latest_checkpoint()
    if not checkpoint:
        return {"warming": False, "reason": "no checkpoint"}

    now = time.monotonic()
    with _warmup_lock:
        if now - _last_warmup.get(checkpoint, 0.0) < WARMUP_MIN_INTERVAL:
            return {"warming": False, "reason": "recently warmed"}
        _last_warmup[checkpoint] = now

    background_tasks.add_task(_warm_sampler, checkpoint)
    return {"warming": True}


@app.post("/api/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Empty input"}, status_code=400)

    checkpoint = req.checkpoint or get_latest_checkpoint()
    if not checkpoint:
        return {
            "error": "No checkpoint yet — training is still in progress.",
            "translation": None,
        }

    # Deterministic phrasebook: an exact full-phrase match on a curated set
    # bypasses the model, for consistency with the on-page glossary and to avoid
    # the empty/lexical failures the model has on these short phrases. Gated to
    # the default checkpoint so the compare dropdown still shows raw model output.
    if checkpoint == get_latest_checkpoint():
        hit = phrasebook_lookup(text, req.direction)
        if hit:
            translation, resolved_dir, detected = hit
            return {
                "translation": translation,
                "direction": resolved_dir,
                "detected_lang": detected,
                "source": "phrasebook",
            }

    direction = req.direction
    if direction == "auto":
        direction = detect_language(text)

    # Use the EXACT instruction the model was trained on — a bare
    # "Translate to English:" / "Translate to Lun Bawang:" with no extra
    # wording (see train_translator.py make_datums). Two earlier embellishments
    # were both out-of-distribution and reliably produced empty completions on
    # hard single words:
    #   1. "Translate this everyday Lun Bawang sentence to English:" — calling a
    #      single word a "sentence" emptied "langub" 5/5 at every reasoning
    #      effort (0/5 with the bare prompt).
    #   2. an appended "(Output only the translation of this word or phrase.)"
    #      hint for short inputs — emptied "water" 5/5 (0/5 without it).
    # No retry ladder can recover these because the emptiness is in the prompt,
    # not the effort. The model already outputs just the translation (it was
    # trained to) and conversational register is steered by the system prompt,
    # so matching training exactly loses nothing and fixes the empties.
    if direction == "lb2en":
        user_content = f"Translate to English:\n{text}"
        detected_lang = "lb"
    else:
        user_content = f"Translate to Lun Bawang:\n{text}"
        detected_lang = "en"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=TINKER_BASE)

        messages_base = [{"role": "system", "content": SYSTEM_PROMPT}]
        is_inkling = checkpoint in _inkling_checkpoint_paths()

        def _completion(content: str, effort: str | None) -> str:
            kwargs = {"reasoning_effort": effort} if effort else {}
            r = client.chat.completions.create(
                model=checkpoint,
                messages=messages_base + [{"role": "user", "content": content}],
                max_tokens=256,
                temperature=0.1,
                top_p=0.9,
                **kwargs,
            )
            return strip_think_tags(r.choices[0].message.content or "")

        # Escalation ladder. reasoning_effort="none" (the effective default
        # when unset) produces single-token empty completions constantly, so
        # "low" is the floor; each retry steps up one rung.
        _NEXT_EFFORT = {"low": "medium", "medium": "high"}

        def _effort_for(source: str) -> str:
            """Starting reasoning effort. "medium" for every input, long or
            short.

            "medium" is the floor because "low" produces single-token empty
            completions at rates that depend on the checkpoint and are not
            worth chasing per-input. On checkpoint-11500, short en->lb phrases
            came back empty 55% of the time at "low" vs 10% at "medium"; on the
            dictionary-retrained v2 weights the low-effort empty rate on single
            Lun Bawang words is worse still (~36% lb->en) even though those are
            exactly the inputs it was trained on. At "medium" both collapse to
            0%. Long inputs rarely empty at either setting, but keeping one
            effort for everything removes a length heuristic that never tracked
            the real (lexical, not length-based) cause of the empties, at the
            cost of a little latency on long inputs — a trade we take for
            consistency and to stay robust across checkpoints.
            """
            return "medium"

        # On an empty completion, retry across these settings in order. The
        # empties are stochastic, effort-dependent, AND come in short
        # time-correlated waves where several settings fail together — so a
        # single medium->high escalation sometimes lands entirely inside a wave
        # and still returns empty (observed live: an input that was 0% empty at
        # medium moments earlier came back empty 8/8 during a wave). Cycling
        # through *different* settings — including "none" (no reasoning_effort),
        # which stayed reliable on the v2 weights while "low"/"medium" were
        # waving — gives uncorrelated attempts that ride the wave out. "low" is
        # skipped: it measured the single worst setting on these weights.
        _EMPTY_RETRY_EFFORTS = ["high", None, "medium", "high"]

        def _call(content: str, source: str) -> str:
            """Translate `content`; `source` chooses the starting effort.

            Returns the first non-empty completion, escalating/cycling settings
            on empty (see _EMPTY_RETRY_EFFORTS). Normal inputs return on the
            first call; only a genuinely wave-affected hard word pays the extra
            round-trips. Only Inkling checkpoints set reasoning_effort at all,
            so none of this applies to Qwen.
            """
            if not is_inkling:
                return _completion(content, None)
            out = _completion(content, _effort_for(source))
            for eff in _EMPTY_RETRY_EFFORTS:
                if out:
                    break
                out = _completion(content, eff)
            return out

        # Whole-sentence translation
        translation = clean_translation(_call(user_content, text), source=text)

        # Clause-by-clause translation (en2lb only, when >= 2 clauses detected)
        result = {
            "translation": translation,
            "direction": direction,
            "detected_lang": detected_lang,
        }

        if direction == "en2lb" and req.clause_split:
            clauses = split_clauses(text)
            if len(clauses) >= 2:
                clause_parts = []
                for clause in clauses:
                    clause_content = f"Translate to Lun Bawang:\n{clause}"
                    clause_parts.append(clean_translation(_call(clause_content, clause)))
                result["clauses"] = clauses
                result["clause_translation"] = ", ".join(clause_parts)

        return result
    except Exception as e:
        return JSONResponse(
            {"error": f"Translation failed: {e}", "translation": None},
            status_code=500,
        )


def _truncate_ip(ip: str | None) -> str | None:
    """Truncate IP to /24 prefix for privacy (1.2.3.4 → 1.2.3.x)."""
    if not ip:
        return ip
    parts = ip.split(".")
    if len(parts) == 4:           # IPv4
        return f"{parts[0]}.{parts[1]}.{parts[2]}.x"
    # IPv6: keep first 3 groups only
    parts = ip.split(":")
    return ":".join(parts[:3]) + ":x" if len(parts) >= 3 else ip


def _push_feedback_to_github(new_row: dict):
    """Append a new feedback row to eval/feedback.csv on GitHub.

    Fetches the existing CSV, inserts the new row (deduped by created_at),
    reassigns sequential ids, and writes back. GitHub is the sole store —
    no local DB or file needed.
    """
    if not GITHUB_TOKEN:
        return
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    cols = ["id", "created_at", "ip_prefix", "user_agent", "source_text",
            "direction", "checkpoint", "model_output", "rating", "correction"]

    # Fetch existing GitHub CSV
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.ok else None

    # Parse existing rows keyed by created_at
    merged: dict = {}
    if r.ok:
        raw = base64.b64decode(r.json()["content"]).decode()
        reader = csv.reader(io.StringIO(raw))
        next(reader, None)  # skip header
        for row in reader:
            if row:
                merged[row[1]] = row  # key by created_at (index 1)

    # Insert new row if not already present (created_at is microsecond-unique)
    ts = new_row["created_at"]
    if ts not in merged:
        merged[ts] = [
            "",  # id assigned below
            ts,
            new_row["ip_prefix"],
            new_row["user_agent"],
            new_row["source_text"],
            new_row["direction"],
            new_row["checkpoint"],
            new_row["model_output"],
            new_row["rating"],
            new_row["correction"] or "",
        ]

    # Sort by timestamp and assign sequential ids
    all_rows = sorted(merged.values(), key=lambda row: row[1])
    for i, row in enumerate(all_rows, 1):
        row[0] = i

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(all_rows)
    content_b64 = base64.b64encode(buf.getvalue().encode()).decode()
    payload = {"message": "auto: update feedback.csv", "content": content_b64}
    if sha:
        payload["sha"] = sha
    requests.put(url, headers=headers, json=payload)


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request, background_tasks: BackgroundTasks):
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
         request.client.host if request.client else None)
    new_row = {
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "ip_prefix":   _truncate_ip(ip),
        "user_agent":  request.headers.get("user-agent", ""),
        "source_text": req.source_text,
        "direction":   req.direction,
        "checkpoint":  req.checkpoint,
        "model_output": req.model_output,
        "rating":      req.rating,
        "correction":  req.correction,
    }
    background_tasks.add_task(_push_feedback_to_github, new_row)
    return {"ok": True}


# Static files — must be mounted last so API routes take priority
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run("serve:app", host=args.host, port=args.port, reload=args.reload)
