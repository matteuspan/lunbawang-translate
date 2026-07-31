"""
Ad-hoc experiment: does raising Inkling's thinking-effort level reduce the
repetition-loop / empty-output failures seen on hard en->lb cases, and at
what time cost? Not wired into training or serving — standalone probe.

Usage: python3.13 eval/effort_experiment.py [checkpoint_path]
"""
import json
import re
import sys
import time
from pathlib import Path

import tinker
from tinker import ServiceClient, SamplingParams, ModelInput
from tml_renderers.chat import MessageList, MessageChannel, Text
from tml_renderers.tinker import token_spans_to_tinker_model_input
from tml_renderers.v0 import Renderer
from tml_renderers.tokenizers import o200k_base_chat

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "tinker_state_inkling.json"

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate ONLY the exact text provided — output just the translation, nothing else. "
    "Use everyday conversational language, not religious or scriptural register. "
    "Proper names (e.g. Bethel, Joyce, Sarah) are names of ordinary people — do not treat them as biblical references. "
    "Preserve proper nouns (personal names, place names) exactly as they appear in the input unless you are certain of the standard equivalent in the target language. "
    "Do not expand, paraphrase, or add any meaning not present in the input. "
    "Do not produce Bible verse language."
)

# ── Test cases pulled from the checkpoint-4000 eval JSONL ──
LONG_REPETITION_CASE = (
    "Crocodile was overjoyed to hear Mouse-deer call him “brother.” “Oh, brother "
    "Mouse-deer, I’m hungry. I want to eat one side of Cow’s thigh, but he’s not "
    "willing to give it to me. He says it’s not fair. He wants to hold a hearing first,” "
    "said Crocodile."
)
LONG_CASE_2 = (
    "“Hear ye, all animals,” said Mouse-deer. “Let us disperse from here; we are "
    "not so vile. We are not foolish like that Abscessed Face. First, he took me as a friend. "
    "He took me as a brother. But after a while, he got bored. He tried to eat me. Is this good? "
    "As for this, Cow helped him, poor Cow, and, oh, dear, he tried to eat a side of Cow’s "
    "thigh. Is that right, do you think?” said Mouse-deer."
)
SHORT_EMPTY_CASES = ["loud sound", "straight ahead", "go over there", "far from here"]

EFFORTS = [0.0, 0.3, 0.6]
TRIALS_PER_CONDITION = 3


def looks_repetitive(text, min_repeats=4):
    """Heuristic: does a run of >=3 words repeat back-to-back min_repeats+ times?"""
    words = text.split()
    for n in (3, 4, 5, 6):
        for i in range(len(words) - n * min_repeats):
            chunk = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == chunk for k in range(min_repeats)):
                return True
    return False


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    if checkpoint_path is None:
        state = json.loads(STATE_FILE.read_text())
        ck = next(c for c in state["checkpoints"] if c["step"] == 4000)
        checkpoint_path = ck["path"]

    print(f"Checkpoint: {checkpoint_path}\n")

    service = ServiceClient()
    sc = service.create_sampling_client(checkpoint_path)
    renderer = Renderer(o200k_base_chat())
    raw_tokenizer = o200k_base_chat()
    params = SamplingParams(max_tokens=256, temperature=0.1, top_p=0.9)

    def translate_at_effort(user_content, effort):
        messages = MessageList.from_oss_messages([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
        t0 = time.monotonic()
        spans, parser = renderer.render_for_completion_with_effort(messages, effort)
        prompt = token_spans_to_tinker_model_input(spans)
        result = sc.sample(prompt, num_samples=1, sampling_params=params).result()
        tokens = list(result.sequences[0].tokens)
        # parser.parse_tokens() only returns *completed* messages — a
        # generation that's still mid-loop when it hits max_tokens (never
        # emits a closing boundary) parses to zero messages, which looks
        # identical to "empty" unless we also inspect the raw decode.
        raw_text = raw_tokenizer.decode(tokens)
        parsed = parser.parse_tokens(tokens)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        text = ""
        for msg in parsed:
            if msg.channel_enum == MessageChannel.Analysis:
                continue
            if isinstance(msg.content, Text):
                text = msg.content.text.strip()
                break
        return text, raw_text, elapsed_ms, len(tokens)

    cases = (
        [("long_repetition", LONG_REPETITION_CASE), ("long_2", LONG_CASE_2)]
        + [(f"short_{s[:12]}", s) for s in SHORT_EMPTY_CASES]
    )

    results = []
    for label, src in cases:
        user_content = f"Translate to Lun Bawang:\n{src}"
        print(f"=== {label} ({len(src.split())} words) ===")
        for effort in EFFORTS:
            for trial in range(TRIALS_PER_CONDITION):
                text, raw_text, ms, ntok = translate_at_effort(user_content, effort)
                repeats = looks_repetitive(raw_text)  # check the RAW stream, not just the parsed answer
                unclosed = ntok >= 250 and not text  # hit the token cap without ever closing a message
                empty = (not text) and not repeats and not unclosed
                if repeats:
                    flag = " REPEAT"
                elif unclosed:
                    flag = " UNCLOSED"
                elif empty:
                    flag = " EMPTY"
                else:
                    flag = ""
                preview = (text or raw_text)[:90] + ("…" if len(text or raw_text) > 90 else "")
                print(f"  effort={effort:<4} trial={trial} {ms:>6}ms {ntok:>4}tok{flag:<10} {preview!r}")
                results.append({
                    "label": label, "effort": effort, "trial": trial,
                    "ms": ms, "tokens": ntok, "empty": empty, "unclosed": unclosed,
                    "repeats": repeats, "text": text, "raw_text": raw_text,
                })
        print()

    out_path = ROOT / "eval" / "effort_experiment_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved -> {out_path}")

    # ── Summary ──
    print("\n=== Summary by effort ===")
    for effort in EFFORTS:
        subset = [r for r in results if r["effort"] == effort]
        n = len(subset)
        n_empty = sum(r["empty"] for r in subset)
        n_repeat = sum(r["repeats"] for r in subset)
        n_unclosed = sum(r["unclosed"] for r in subset)
        n_bad = sum(r["empty"] or r["repeats"] or r["unclosed"] for r in subset)
        avg_ms = sum(r["ms"] for r in subset) / n
        print(f"  effort={effort:<4} n={n}  bad={n_bad}/{n}  (empty={n_empty} repeat={n_repeat} unclosed={n_unclosed})  avg_ms={avg_ms:.0f}")


if __name__ == "__main__":
    main()
