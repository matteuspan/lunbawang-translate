"""
OCR the paper Lun Bawang -> English dictionary into structured entries.

Stage B of the dictionary pipeline (Stage A is imaging the pages, Stage C is
build_dictionary_corpus.py). Each page image is sent to a Claude vision model
with a *strict* tool schema, so the model must return well-formed structured
entries rather than free text. Output is one JSON object per page in a JSONL
file, ready for the post-processor.

Design notes
------------
* Faithful transcription. The system prompt tells the model to copy the
  orthography exactly (glottal-stop apostrophes, subscripts, angle-bracket
  cross-refs) and never to normalise spelling or invent entries — the main
  risk of using an LLM as an OCR engine on a low-resource language.
* Strict structured output. A single forced tool call (`record_page`,
  strict=True) guarantees the shape below; thinking is disabled because forced
  tool use requires it and this is a perception task, not a reasoning one.
* Batch by default. 400 pages aren't latency-sensitive, so the default path
  uses the Batch API (~50% cheaper). `--sample N` runs the first N pages
  synchronously instead, for validating quality and tuning the prompt before
  committing to the whole book.
* Resumable. Pages already present in the output JSONL are skipped, so an
  interrupted run (or a re-run after adding pages) only does the new work.

Usage
-----
  pip install anthropic
  export ANTHROPIC_API_KEY=...            # or `ant auth login`

  # 1) validate on a handful of real scans first (synchronous, fast feedback)
  python3 ocr_dictionary.py pages/ --sample 5

  # 2) once quality looks right, run the whole book through the Batch API
  python3 ocr_dictionary.py pages/ --out dictionary_entries.jsonl

`pages/` is a directory of page images (`.jpg`/`.jpeg`/`.png`/`.webp`),
processed in sorted filename order — so name them zero-padded (page_0001.jpg).
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

MODEL_DEFAULT = "claude-sonnet-5"   # strong vision, cheaper than Opus; --model to change
# A dense four-column spread can hold ~140 entries with example sentences, which
# is well over 20k output tokens of JSON — an 8k cap silently truncates the
# forced tool call and the page parses to nothing. Give it real headroom.
MAX_TOKENS = 32000

IMAGE_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".png": "image/png", ".webp": "image/webp"}

SYSTEM_PROMPT = (
    "You are a meticulous OCR transcriber for a printed Lun Bawang - English "
    "dictionary from Borneo. Lun Bawang is a low-resource language, so accuracy "
    "of the exact printed forms matters more than anything.\n"
    "Rules:\n"
    "- Transcribe EXACTLY what is printed. Never correct, modernise, or "
    "normalise spelling. Preserve glottal-stop apostrophes (' and the typographic "
    "form) and every diacritic as printed.\n"
    "- Each image is a screenshot from an e-book reader and usually shows a "
    "two-page spread. Ignore everything that is not dictionary text: reader "
    "buttons and toolbars, the page-number slider, sidebar icons, running "
    "headers, page numbers, the large section-letter heading (e.g. 'Aa'), and "
    "any faint show-through from the reverse side. If a page of the spread is "
    "blank or front-matter, skip it.\n"
    "- Read in natural order: do the left page before the right page, and within "
    "each page finish the entire left column top-to-bottom before the right "
    "column.\n"
    "- Each bold head-word begins an entry. A trailing subscript digit "
    "(e.g. mekafal with a small 1) distinguishes homographs — record it in "
    "`homograph`, and put the bare word (no digit) in `headword`.\n"
    "- Numbered senses (1. ... 2. ...) become separate items in `senses` with "
    "their number in `n`; an unnumbered single gloss is one sense with n = null.\n"
    "- Decide each sense's `kind`: 'equivalent' when the gloss is a direct "
    "word/phrase translation (e.g. 'left-handed', 'water', 'good morning'); "
    "'definition' when it is a descriptive gloss (e.g. 'can be channeled or "
    "drained', 'able to make a partition or wall').\n"
    "- 'also X' forms go in `variants`. Cross-references printed in <angle "
    "brackets> go in `cross_refs`, writing any subscript as a trailing digit "
    "(<abang with small 1> -> 'abang1').\n"
    "- A register label printed with a sense (e.g. 'archaic', 'vulgar') goes in "
    "`register`, not in the gloss.\n"
    "- Many entries include one or more EXAMPLE SENTENCES: an italic Lun Bawang "
    "sentence immediately followed by its English translation in roman type. "
    "Capture each as an item in `examples`, with the Lun Bawang sentence in "
    "`lun_bawang` and its English translation in `english`, both verbatim. An "
    "entry with no example sentence has an empty `examples` array.\n"
    "- Do NOT invent entries. If the very first or very last line is a partial "
    "entry cut off by the page edge, transcribe what is visible.\n"
    "- If the spread contains no dictionary entries at all, return an empty "
    "`entries` array."
)

# Strict schema — the model is forced to call this tool, so output always matches.
ENTRY_TOOL = {
    "name": "record_page",
    "description": "Record every dictionary entry transcribed from this page image.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["entries"],
        "properties": {
            "entries": {
                "type": "array",
                "description": "All entries on the page, in reading order.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["headword", "homograph", "senses", "examples", "variants", "cross_refs"],
                    "properties": {
                        "headword": {
                            "type": "string",
                            "description": "The Lun Bawang head-word exactly as printed, WITHOUT any subscript digit.",
                        },
                        "homograph": {
                            "type": ["integer", "null"],
                            "description": "The subscript digit on the head-word, or null if there is none.",
                        },
                        "senses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["n", "gloss", "kind", "register"],
                                "properties": {
                                    "n": {
                                        "type": ["integer", "null"],
                                        "description": "The sense number if the entry is numbered, else null.",
                                    },
                                    "gloss": {
                                        "type": "string",
                                        "description": "The English gloss for this sense, exactly as printed (minus any register label).",
                                    },
                                    "kind": {
                                        "type": "string",
                                        "enum": ["equivalent", "definition"],
                                        "description": "'equivalent' = direct translation; 'definition' = descriptive gloss.",
                                    },
                                    "register": {
                                        "type": ["string", "null"],
                                        "description": "A usage label such as 'archaic' printed with the sense, else null.",
                                    },
                                },
                            },
                        },
                        "examples": {
                            "type": "array",
                            "description": "Example sentences: an italic Lun Bawang sentence and its English translation.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["lun_bawang", "english"],
                                "properties": {
                                    "lun_bawang": {
                                        "type": "string",
                                        "description": "The Lun Bawang example sentence, exactly as printed.",
                                    },
                                    "english": {
                                        "type": "string",
                                        "description": "Its English translation, exactly as printed.",
                                    },
                                },
                            },
                        },
                        "variants": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative surface forms printed as 'also X'.",
                        },
                        "cross_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Root cross-references printed in <angle brackets>, subscripts as trailing digits.",
                        },
                    },
                },
            }
        },
    },
}


def image_block(path: Path) -> dict:
    media_type = IMAGE_EXTS[path.suffix.lower()]
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data}}


def build_params(path: Path, model: str) -> dict:
    """The Messages-API params for one page — shared by sync and batch paths."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # Forced tool use requires thinking off; extraction needs no reasoning.
        "thinking": {"type": "disabled"},
        "system": [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},   # stable prefix, cached across pages
        }],
        "tools": [ENTRY_TOOL],
        "tool_choice": {"type": "tool", "name": "record_page"},
        "messages": [{
            "role": "user",
            "content": [
                image_block(path),
                {"type": "text", "text": "Transcribe every entry on this dictionary page."},
            ],
        }],
    }


def extract_entries(message) -> list:
    """Pull the tool-call input out of a Message response."""
    for block in message.content:
        if block.type == "tool_use" and block.name == "record_page":
            return block.input.get("entries", [])
    return []


def load_done(out_path: Path) -> set:
    """Image names already recorded in the output JSONL (for resume)."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["image"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def list_pages(images_dir: Path) -> list:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run_sample(client, pages, model, out_path):
    """Synchronous path — immediate feedback for validating a few pages.

    Streams because a dense spread can emit tens of thousands of output tokens,
    which would otherwise risk an HTTP timeout on a non-streaming call."""
    with out_path.open("a", encoding="utf-8") as f:
        for path in pages:
            print(f"  {path.name} ... ", end="", flush=True)
            try:
                with client.messages.stream(**build_params(path, model)) as stream:
                    msg = stream.get_final_message()
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"ERROR: {e}")
                continue
            if msg.stop_reason == "refusal":
                print("REFUSED by safety classifier")
                continue
            entries = extract_entries(msg)
            note = ""
            if msg.stop_reason == "max_tokens":
                note = "  ** TRUNCATED (hit max_tokens) — raise MAX_TOKENS **"
            f.write(json.dumps({"image": path.name, "entries": entries},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"{len(entries)} entries{note}")


def run_batch(client, pages, model, out_path, poll_interval):
    """Batch API path — ~50% cheaper, for the full book."""
    requests = [{"custom_id": path.name, "params": build_params(path, model)}
                for path in pages]
    print(f"Submitting batch of {len(requests)} pages ...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch id: {batch.id}")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"  status={batch.processing_status} "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored}", flush=True)
        time.sleep(poll_interval)

    ok = err = truncated = 0
    with out_path.open("a", encoding="utf-8") as f:
        for item in client.messages.batches.results(batch.id):
            if item.result.type != "succeeded":
                err += 1
                print(f"  {item.custom_id}: {item.result.type}")
                continue
            msg = item.result.message
            if msg.stop_reason == "max_tokens":
                truncated += 1
                print(f"  {item.custom_id}: ** TRUNCATED (hit max_tokens) **")
            entries = extract_entries(msg)
            f.write(json.dumps({"image": item.custom_id, "entries": entries},
                               ensure_ascii=False) + "\n")
            ok += 1
    msg = f"Done: {ok} pages written, {err} failed."
    if truncated:
        msg += f" {truncated} TRUNCATED — raise MAX_TOKENS and re-run those pages."
    print(msg + " Re-run to retry failures.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir", type=Path, help="Directory of page images.")
    ap.add_argument("--out", type=Path, default=Path("dictionary_entries.jsonl"),
                    help="Output JSONL (default: dictionary_entries.jsonl).")
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help=f"Claude model id (default: {MODEL_DEFAULT}).")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="Run the first N pages synchronously instead of batching.")
    ap.add_argument("--max-pages", type=int,
                    help="Cap the number of pages processed (after resume filtering).")
    ap.add_argument("--poll-interval", type=float, default=30.0,
                    help="Seconds between batch status polls (default: 30).")
    args = ap.parse_args()

    if not args.images_dir.is_dir():
        sys.exit(f"Not a directory: {args.images_dir}")

    try:
        import anthropic
    except ImportError:
        sys.exit("The anthropic SDK is required: pip install anthropic")

    client = anthropic.Anthropic()   # resolves ANTHROPIC_API_KEY or an `ant` profile

    pages = list_pages(args.images_dir)
    if not pages:
        sys.exit(f"No images ({'/'.join(IMAGE_EXTS)}) found in {args.images_dir}")

    done = load_done(args.out)
    todo = [p for p in pages if p.name not in done]
    if done:
        print(f"Resuming: {len(done)} pages already done, {len(todo)} remaining.")

    if args.sample:
        todo = todo[:args.sample]
    elif args.max_pages:
        todo = todo[:args.max_pages]

    if not todo:
        print("Nothing to do — all pages already processed.")
        return

    print(f"Model: {args.model}   Pages this run: {len(todo)}   Output: {args.out}")
    if args.sample:
        run_sample(client, todo, args.model, args.out)
    else:
        run_batch(client, todo, args.model, args.out, args.poll_interval)


if __name__ == "__main__":
    main()
