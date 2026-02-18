"""
Web server for the Lun Bawang ↔ English translator.

Usage:
  python3.13 serve.py              # runs on http://localhost:8000
  python3.13 serve.py --port 8080  # custom port

The backend reads tinker_state.json to find the latest checkpoint and
calls the Tinker OpenAI-compatible serving API for inference.
"""

import json
import os
import re
import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────

STATE_FILE  = Path(__file__).parent / "tinker_state.json"
STATIC_DIR  = Path(__file__).parent / "static"
API_KEY     = os.environ["TINKER_API_KEY"]
TINKER_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate accurately and naturally."
)

# Common English function/content words for language auto-detection
ENGLISH_WORDS = {
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
    "been", "am", "god", "lord", "said", "shall", "did", "does", "been",
    "i", "those", "through", "before", "after", "many", "also", "where",
    "much", "must", "upon", "shall", "great", "against", "between", "down",
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


def strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>…</think> reasoning blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def get_latest_checkpoint() -> str | None:
    checkpoints = get_state().get("checkpoints", [])
    return checkpoints[-1]["path"] if checkpoints else None


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(title="LunBawang Translate")


class TranslateRequest(BaseModel):
    text: str
    direction: str = "auto"  # "auto" | "lb2en" | "en2lb"


@app.get("/api/status")
def status():
    state = get_state()
    checkpoint = get_latest_checkpoint()
    return {
        "ready": checkpoint is not None,
        "checkpoint": checkpoint,
        "steps": state.get("steps", 0),
        "num_checkpoints": len(state.get("checkpoints", [])),
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Empty input"}, status_code=400)

    checkpoint = get_latest_checkpoint()
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
        detected_lang = "lb"
    else:
        user_content = f"Translate to Lun Bawang:\n{text}"
        detected_lang = "en"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=TINKER_BASE)
        response = client.chat.completions.create(
            model=checkpoint,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
        )
        translation = strip_think_tags(response.choices[0].message.content)
        return {
            "translation": translation,
            "direction": direction,
            "detected_lang": detected_lang,
        }
    except Exception as e:
        return JSONResponse(
            {"error": f"Translation failed: {e}", "translation": None},
            status_code=500,
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
