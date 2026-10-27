"""Baselines: run one model over the held-out gold RFPs, no fine-tuning.

Every model gets the SAME input (the page-marked .txt), the SAME instructions
(prompts.EXTRACTION_SYSTEM), and the SAME single worked example. The only
variable is the model. Predictions land in out/baselines/<alias>-<rfp_id>.json
and raw model text in out/baselines/raw/ for post-mortems.

Backends:
  claude    Anthropic API (claude-opus-4-8, structured output), the bar to beat.
            Needs ANTHROPIC_API_KEY in .env.
  qwen      Qwen/Qwen3.5-4B          un-tuned, chunk-and-merge (small model)
  kimi      moonshotai/Kimi-K2.6     un-tuned, chunk-and-merge (32k window)
  inkling   thinkingmachines/Inkling un-tuned, whole doc (64k window)
            Tinker backends need tinker + tinker_cookbook (see README).
  glm       GLM 5.2 over the Z.ai API (needs ZAI_API_KEY in .env)

Usage: python src/baseline.py <backend> <rfp_id> [<rfp_id> ...]
       python src/baseline.py <backend> --gold      # all three gold RFPs
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from prompts import (EXTRACTION_SYSTEM, EXTRACTION_USER,
                     FEW_SHOT_ASSISTANT, FEW_SHOT_USER)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "baselines"
GOLD_IDS = ["3202600219", "HHS0015558v2", "3202600177"]

TINKER_MODELS = {
    "qwen": "Qwen/Qwen3.5-4B",
    "kimi": "moonshotai/Kimi-K2.6",
    "inkling": "thinkingmachines/Inkling",
}
CLAUDE_MODEL = "claude-opus-4-8"
GLM_MODEL = "glm-5.2"
MAX_OUT_TOKENS = 48000
# Chunk-and-merge, per backend. Qwen because a 4B model does better on small
# pieces; Kimi because its Tinker serving window is 32,768 tokens total, so a
# whole big contract plus its answer can never fit. Inkling (65,536) takes the
# whole document; its answer budget is clamped in TinkerSampler.sample so
# prompt + answer always fits the window. Tuned checkpoints (any round) all
# chunk at TRAIN_CHUNK_WORDS — they were taught on 4500-word pieces, so they
# read the contracts the same way at eval time (see run_tinker).
CHUNK_WORDS = {"qwen": 4500, "kimi": 4500}
TRAIN_CHUNK_WORDS = 4500


def tuned_base(backend: str) -> str | None:
    """'qwen-tuned' -> 'qwen'; None when the backend isn't a tuned alias.

    Valid aliases are the round-1 contestant names plus anything recorded in
    out/checkpoints.json by train_lora.py (round-2 aliases like 'qwen2').
    """
    if not backend.endswith("-tuned"):
        return None
    base = backend.removesuffix("-tuned")
    if base in TINKER_MODELS:
        return base
    reg_path = ROOT / "out" / "checkpoints.json"
    if reg_path.exists() and base in json.loads(reg_path.read_text()):
        return base
    return None


def load_env() -> None:
    """Read key=value lines from this repo's .env."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def rfp_text(rfp_id: str) -> str:
    path = ROOT / "data" / "rfps" / f"{rfp_id}.txt"
    if not path.exists():
        sys.exit(f"no text for {rfp_id} at {path}")
    return path.read_text()


def split_pages(text: str) -> list[tuple[int, str]]:
    """[(page_number, page_text), ...] from a page-marked txt."""
    pages: list[tuple[int, list[str]]] = []
    current = 1
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\[page (\d+)\]\s*$", line)
        if m:
            if buf:
                pages.append((current, buf))
            current, buf = int(m.group(1)), []
        else:
            buf.append(line)
    if buf:
        pages.append((current, buf))
    return [(p, "\n".join(ls)) for p, ls in pages]


def chunk_pages(text: str, max_words: int) -> list[str]:
    """Greedy page-boundary chunks, each re-carrying its [page N] markers."""
    chunks: list[list[str]] = [[]]
    words = 0
    for page, body in split_pages(text):
        w = len(body.split())
        if chunks[-1] and words + w > max_words:
            chunks.append([])
            words = 0
        chunks[-1].append(f"[page {page}]\n{body}")
        words += w
    return ["\n".join(c) for c in chunks if c]


def user_turn(rfp_id: str, text: str) -> str:
    return EXTRACTION_USER.format(rfp_id=rfp_id) + "\n\nRFP TEXT:\n" + text


# --- lenient JSON recovery for un-tuned open models -------------------------
def parse_matrix_text(raw: str, rfp_id: str) -> tuple[dict, str]:
    """Best-effort parse of a model's text into a matrix dict.

    Returns (matrix, note). On total failure the matrix has zero requirements
    and the note says why — an unparseable answer scores as it deserves.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    # Candidate start positions for the answer object. Un-tuned models often
    # "think out loud" first, and that prose can contain stray braces — so
    # prefer positions that look like the matrix itself ({"rfp_id" ... or
    # {"requirements" ...), trying the LAST one first (the answer follows the
    # thinking). Fall back to the first brace/bracket in the text.
    anchored = [m.start() for m in
                re.finditer(r'\{\s*"(?:rfp_id|requirements)"', text)]
    first = min((i for i in (text.find("{"), text.find("[")) if i != -1),
                default=-1)
    starts = list(reversed(anchored)) + ([first] if first != -1 else [])
    if not starts:
        return {"rfp_id": rfp_id, "requirements": []}, "no JSON found"

    decoder = json.JSONDecoder()
    obj, note = None, "ok"
    for start in starts:
        cand = text[start:]
        try:
            obj, _ = decoder.raw_decode(cand)
            break
        except json.JSONDecodeError:
            # truncated output: cut to the last complete row, close the JSON
            last = cand.rfind("},")
            if last != -1:
                for tail in ("]}", "]"):
                    try:
                        obj, _ = decoder.raw_decode(cand[: last + 1] + tail)
                        note = "repaired truncated JSON"
                        break
                    except json.JSONDecodeError:
                        continue
            if obj is not None:
                break
    if obj is None:
        return {"rfp_id": rfp_id, "requirements": []}, "unparseable JSON"
    if isinstance(obj, list):
        obj = {"requirements": obj}
    reqs = obj.get("requirements")
    if not isinstance(reqs, list):
        return {"rfp_id": rfp_id, "requirements": []}, "no requirements array"
    rows = [r for r in reqs if isinstance(r, dict)]
    return {"rfp_id": rfp_id, "requirements": rows}, note


def merge_chunks(parts: list[dict], rfp_id: str) -> dict:
    """Concatenate chunk rows, dropping exact repeats.

    Two rows are the same requirement when their section and text match after
    stripping everything but letters/digits. Applied to every chunked backend
    identically (fairness rule) — it only ever removes a model's self-repeats,
    since a duplicate can never match a second gold row (each gold row is
    claimed once by the scorer).
    """
    rows, seen, dropped = [], set(), 0
    for part in parts:
        for r in part["requirements"]:
            key = (re.sub(r"[^a-z0-9]", "", str(r.get("section_ref", "")).lower()),
                   re.sub(r"[^a-z0-9]", "", str(r.get("verbatim_text", "")).lower()))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            rows.append(r)
    if dropped:
        print(f"  [dedupe: dropped {dropped} repeated rows]")
    for i, r in enumerate(rows, 1):
        r["id"] = f"R-{i:03d}"
    return {"rfp_id": rfp_id, "requirements": rows}


# --- backends ---------------------------------------------------------------
def run_claude(rfp_id: str) -> tuple[dict, str]:
    from anthropic import Anthropic
    from teacher import strict

    schema = strict(json.loads((ROOT / "schema.json").read_text()))
    client = Anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system=EXTRACTION_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[
            {"role": "user", "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_turn(rfp_id, rfp_text(rfp_id))},
        ],
    ) as stream:
        message = stream.get_final_message()
    raw = "".join(b.text for b in message.content if b.type == "text")
    matrix = json.loads(raw)
    matrix["rfp_id"] = rfp_id
    return matrix, raw


def run_glm(rfp_id: str) -> tuple[dict, str]:
    if not os.environ.get("ZAI_API_KEY"):
        sys.exit("GLM baseline needs ZAI_API_KEY in .env (get one at z.ai), "
                 "then rerun: python src/baseline.py glm --gold")
    import httpx

    resp = httpx.post(
        "https://api.z.ai/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['ZAI_API_KEY']}"},
        json={
            "model": GLM_MODEL,
            "max_tokens": MAX_OUT_TOKENS,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": FEW_SHOT_USER},
                {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                {"role": "user",
                 "content": user_turn(rfp_id, rfp_text(rfp_id))},
            ],
        },
        timeout=1800,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    matrix, note = parse_matrix_text(raw, rfp_id)
    if note != "ok":
        print(f"  [glm parse: {note}]")
    return matrix, raw


class TinkerSampler:
    def __init__(self, backend: str):
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.types = types
        base = tuned_base(backend)
        if base:
            # Fine-tuned checkpoint: same renderer it was trained with,
            # sampler weights from the path train_lora.py recorded.
            reg = json.loads((ROOT / "out" / "checkpoints.json").read_text())
            entry = reg[base]
            base_model = entry["base_model"]
            renderer_name = entry["renderer"]
            model_path = entry["sampler_path"]
        else:
            base_model = TINKER_MODELS[backend]
            renderer_name = model_info.get_recommended_renderer_name(base_model)
            model_path = None
        tokenizer = get_tokenizer(base_model)
        self.tokenizer = tokenizer
        self.renderer = renderers.get_renderer(renderer_name, tokenizer)
        service = tinker.ServiceClient()
        self.client = (service.create_sampling_client(model_path=model_path)
                       if model_path else
                       service.create_sampling_client(base_model=base_model))
        # LoRA checkpoints serve on the (bigger-window) peft tier when the
        # capabilities list has one for this base model.
        caps = {m.model_name: m.max_context_length
                for m in service.get_server_capabilities().supported_models}
        # peft tiers are listed as e.g. "moonshotai/Kimi-K2.6:peft:131072"
        peft = max((v for k, v in caps.items()
                    if k.startswith(f"{base_model}:peft")), default=None)
        self.max_context = (peft if model_path and peft else caps[base_model])
        print(f"[{backend}] {base_model}"
              f"{' @ ' + model_path if model_path else ''} "
              f"renderer={renderer_name} context={self.max_context}")

    def sample(self, conversation: list[dict], temperature: float = 0.2,
               seed: int | None = None, max_tokens: int = MAX_OUT_TOKENS) -> str:
        prompt = self.renderer.build_generation_prompt(conversation)
        # Never ask for more answer tokens than the window can hold on top of
        # the prompt — overflow makes the server return an empty sample.
        room = self.max_context - prompt.length - 16
        if room < 1000:
            raise ValueError(
                f"prompt ({prompt.length} tokens) leaves only {room} answer "
                f"tokens in a {self.max_context} window — chunk this backend")
        extra = {"seed": seed} if seed is not None else {}
        params = self.types.SamplingParams(
            max_tokens=min(max_tokens, room), temperature=temperature,
            stop=self.renderer.get_stop_sequences(), **extra)
        response = self.client.sample(
            prompt=prompt, num_samples=1, sampling_params=params).result()
        # Decode the tokens directly rather than trusting the renderer's
        # reply parsing — a truncated or thinking-only reply can otherwise
        # come back as silently empty, and we always want the raw text for
        # post-mortems. parse_matrix_text digs the answer out of any prose.
        return self.tokenizer.decode(response.sequences[0].tokens)


# --- degeneration guard -----------------------------------------------------
# Tinker's SamplingParams has no repetition penalty, so runaway repetition
# (round 1: Inkling fabricating "Exhibit BBBBB..." rows until the output
# budget ran out) is caught here instead: detect, then resample hotter with a
# fresh seed. Harness-level and identical for every Tinker backend, tuned or
# not — no contestant gets a knob the others lack.
DEGEN_RETRIES = 2
LETTER_RUN_RE = re.compile(r"([A-Za-z])\1{15,}")


def degenerate_reason(raw: str, matrix: dict) -> str | None:
    if LETTER_RUN_RE.search(raw):
        return "letter-run repetition"
    rows = matrix["requirements"]
    if len(rows) >= 20:
        texts = [re.sub(r"\s+", " ", str(r.get("verbatim_text", ""))).lower()
                 for r in rows]
        dup = len(texts) - len(set(texts))
        if dup / len(texts) > 0.3:
            return f"{dup}/{len(texts)} rows are repeats"
    return None


def run_tinker(backend: str, rfp_id: str, sampler: TinkerSampler) -> tuple[dict, str]:
    text = rfp_text(rfp_id)
    chunk_words = CHUNK_WORDS.get(
        backend, TRAIN_CHUNK_WORDS if tuned_base(backend) else None)
    chunks = chunk_pages(text, chunk_words) if chunk_words else [text]
    parts, raws = [], []
    for n, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"  chunk {n}/{len(chunks)} ({len(chunk.split())} words)")
        # Tuned checkpoints were trained without the worked example (the
        # fine-tune replaces it), so they're sampled without it too.
        few_shot = ([] if tuned_base(backend) else
                    [{"role": "user", "content": FEW_SHOT_USER},
                     {"role": "assistant", "content": FEW_SHOT_ASSISTANT}])
        conversation = [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            *few_shot,
            {"role": "user", "content": user_turn(rfp_id, chunk)},
        ]
        raw = self_heal_sample(sampler, conversation)
        matrix, note = parse_matrix_text(raw, rfp_id)
        reason = degenerate_reason(raw, matrix)
        for retry in range(1, DEGEN_RETRIES + 1):
            if not reason:
                break
            print(f"  [{backend} degenerate: {reason} — "
                  f"resample {retry}/{DEGEN_RETRIES} at temp 0.7]")
            raw = self_heal_sample(sampler, conversation,
                                   temperature=0.7, seed=retry)
            matrix, note = parse_matrix_text(raw, rfp_id)
            reason = degenerate_reason(raw, matrix)
        if reason:
            print(f"  [{backend} still degenerate ({reason}) — "
                  f"keeping last sample; merge dedupe will trim repeats]")
        raws.append(raw)
        if note != "ok":
            print(f"  [{backend} parse: {note}]")
        parts.append(matrix)
    return merge_chunks(parts, rfp_id), "\n\n=== CHUNK BREAK ===\n\n".join(raws)


def self_heal_sample(sampler: TinkerSampler, conversation: list[dict],
                     tries: int = 3, temperature: float = 0.2,
                     seed: int | None = None) -> str:
    last = None
    for attempt in range(1, tries + 1):
        try:
            return sampler.sample(conversation, temperature=temperature,
                                  seed=seed)
        except Exception as e:  # transient sampling-service hiccups
            last = e
            print(f"  [sample attempt {attempt} failed: {type(e).__name__}]")
    raise last


def main() -> None:
    load_env()
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    backend = sys.argv[1]
    rfp_ids = GOLD_IDS if sys.argv[2] == "--gold" else sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)

    is_tinker = backend in TINKER_MODELS or tuned_base(backend)
    sampler = TinkerSampler(backend) if is_tinker else None
    for rfp_id in rfp_ids:
        print(f"{backend} <- {rfp_id}")
        if backend == "claude":
            matrix, raw = run_claude(rfp_id)
        elif backend == "glm":
            matrix, raw = run_glm(rfp_id)
        elif is_tinker:
            matrix, raw = run_tinker(backend, rfp_id, sampler)
        else:
            sys.exit(f"unknown backend {backend!r} "
                     f"(claude | glm | {' | '.join(TINKER_MODELS)} "
                     f"| <alias>-tuned)")
        pred_path = OUT / f"{backend}-{rfp_id}.json"
        pred_path.write_text(json.dumps(matrix, indent=2))
        (OUT / "raw" / f"{backend}-{rfp_id}.txt").write_text(raw)
        print(f"  {len(matrix['requirements'])} requirements -> "
              f"{pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
