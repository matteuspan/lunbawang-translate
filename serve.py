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
import sqlite3
import argparse
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────

STATE_FILE      = Path(__file__).parent / "tinker_state.json"
STATE_FILE_RUN1 = Path(__file__).parent / "tinker_state_run1.json"
STATIC_DIR  = Path(__file__).parent / "static"
API_KEY     = os.environ["TINKER_API_KEY"]
TINKER_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

FEEDBACK_DIR   = Path(os.environ.get("FEEDBACK_DIR", str(Path(__file__).parent)))
FEEDBACK_DB    = FEEDBACK_DIR / "feedback.db"
FEEDBACK_JSONL = FEEDBACK_DIR / "feedback.jsonl"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO  = "matteuspan/lunbawang-translate"
GITHUB_PATH  = "feedback.csv"

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate ONLY the exact text provided — output just the translation, nothing else. "
    "Do not add Bible verse titles, context, or anything not present in the input."
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


# ── Feedback DB ────────────────────────────────────────────────────────────

def init_feedback_db():
    con = sqlite3.connect(FEEDBACK_DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    NOT NULL,
            ip_address   TEXT,
            user_agent   TEXT,
            source_text  TEXT    NOT NULL,
            direction    TEXT    NOT NULL,
            checkpoint   TEXT    NOT NULL,
            model_output TEXT    NOT NULL,
            rating       INTEGER NOT NULL,
            correction   TEXT
        )
    """)
    con.commit()
    con.close()

init_feedback_db()

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
    """Return all checkpoints across current run + run1 fallback, newest first."""
    current = get_state().get("checkpoints", [])
    if current:
        return current
    # New run has no checkpoints yet — fall back to the previous run
    if STATE_FILE_RUN1.exists():
        return json.loads(STATE_FILE_RUN1.read_text()).get("checkpoints", [])
    return []


def get_latest_checkpoint() -> str | None:
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
    main_state = get_state()
    main_model_id = main_state.get("model_id", "")
    main_ckpts = main_state.get("checkpoints", [])

    run1_state = {}
    if STATE_FILE_RUN1.exists():
        run1_state = json.loads(STATE_FILE_RUN1.read_text())
    run1_model_id = run1_state.get("model_id", "")
    run1_ckpts = run1_state.get("checkpoints", [])

    # If the main state file has the same experiment as run1 (e.g. on Render where
    # tinker_state.json is the committed copy of the old run), treat it as run1.
    if main_model_id and main_model_id == run1_model_id:
        # Merge and deduplicate by step
        merged = {ck["step"]: ck for ck in run1_ckpts}
        merged.update({ck["step"]: ck for ck in main_ckpts})
        run1_ckpts = sorted(merged.values(), key=lambda x: x["step"])
        main_ckpts = []  # nothing left for run2

    result = []

    for ck in run1_ckpts:
        label = f"Run 1 · Step {ck['step']:,}"
        if "epoch" in ck:
            label += f" · Epoch {ck['epoch']}"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    for i, ck in enumerate(main_ckpts):
        label = f"Run 2 · Step {ck['step']:,}"
        if "epoch" in ck:
            label += f" · Epoch {ck['epoch']}"
        if i == len(main_ckpts) - 1:
            label += " (latest)"
        result.append({"label": label, "path": ck["path"], "step": ck["step"]})

    return result


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

    direction = req.direction
    if direction == "auto":
        direction = detect_language(text)

    if direction == "lb2en":
        user_content = f"Translate to English:\n{text}"
        if len(text.split()) <= 5:
            user_content += "\n(Output only the translation of this word or phrase.)"
        detected_lang = "lb"
    else:
        user_content = f"Translate to Lun Bawang:\n{text}"
        detected_lang = "en"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=TINKER_BASE)

        def _call(content: str) -> str:
            r = client.chat.completions.create(
                model=checkpoint,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": content},
                ],
                max_tokens=256,
                temperature=0.1,
                top_p=0.9,
            )
            return strip_think_tags(r.choices[0].message.content)

        # Whole-sentence translation
        translation = clean_translation(_call(user_content), source=text)

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
                    if len(clause.split()) <= 5:
                        clause_content += "\n(Output only the translation of this word or phrase.)"
                    clause_parts.append(clean_translation(_call(clause_content)))
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


def _push_feedback_to_github():
    """Commit feedback.csv to GitHub. IPs are truncated to /24 for privacy."""
    if not GITHUB_TOKEN:
        return
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    cols = ["id", "created_at", "ip_prefix", "user_agent", "source_text",
            "direction", "checkpoint", "model_output", "rating", "correction"]
    con = sqlite3.connect(FEEDBACK_DB)
    rows = con.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    con.close()
    # Replace full ip_address (col index 2) with truncated prefix
    db_cols = ["id", "created_at", "ip_address", "user_agent", "source_text",
               "direction", "checkpoint", "model_output", "rating", "correction"]
    ip_idx = db_cols.index("ip_address")
    rows = [tuple(
        _truncate_ip(v) if i == ip_idx else v
        for i, v in enumerate(row)
    ) for row in rows]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(rows)
    content_b64 = base64.b64encode(buf.getvalue().encode()).decode()
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.ok else None
    payload = {"message": "auto: update feedback.csv", "content": content_b64}
    if sha:
        payload["sha"] = sha
    requests.put(url, headers=headers, json=payload)


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request, background_tasks: BackgroundTasks):
    from datetime import datetime, timezone
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
         request.client.host if request.client else None)
    ua = request.headers.get("user-agent", "")
    created_at = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(FEEDBACK_DB)
    cur = con.execute(
        "INSERT INTO feedback "
        "(created_at,ip_address,user_agent,source_text,direction,checkpoint,model_output,rating,correction) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (created_at, ip, ua, req.source_text, req.direction, req.checkpoint,
         req.model_output, req.rating, req.correction),
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    with open(FEEDBACK_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": row_id, "created_at": created_at, "ip_address": ip,
            "user_agent": ua, "source_text": req.source_text,
            "direction": req.direction, "checkpoint": req.checkpoint,
            "model_output": req.model_output, "rating": req.rating,
            "correction": req.correction,
        }) + "\n")
    background_tasks.add_task(_push_feedback_to_github)
    return {"ok": True}


@app.get("/api/feedback/export")
def export_feedback():
    cols = ["id", "created_at", "ip_address", "user_agent", "source_text",
            "direction", "checkpoint", "model_output", "rating", "correction"]
    con = sqlite3.connect(FEEDBACK_DB)
    rows = con.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    con.close()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerows(rows)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback.csv"},
    )


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
