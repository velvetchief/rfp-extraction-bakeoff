# RFP requirement extraction: Claude, fine-tuned open models, and Jev

This repo compares models on one task. The task is to read a government RFP and list every requirement the bidder must meet, with its section, its exact text, a type, a category and any due date. A bid team calls that list a compliance matrix.

The full write-up is at [prithvidatla.com/evals/jev](https://prithvidatla.com/evals/jev/).

## The data

All documents are public Texas RFPs from the state's Electronic State Business Daily (ESBD). There are 14 of them in `data/rfps/`, as the original PDF and as page-marked text.

- **3 held-out contracts** (`3202600219`, `HHS0015558v2`, `3202600177`) have hand-checked answer keys in `data/gold/`. They hold 54, 181 and 203 requirements. No model trains on them, and the training scripts assert this.
- **11 practice contracts** have teacher labels in `data/train/` and `data/train2/`. The fine-tuning pairs are in `data/pairs/`.

## How a prediction is scored

`evals/score.py` is deterministic. A predicted row matches a gold row when the section numbers agree after normalization and the text overlap (token F1) is at least 0.6. Each gold row can be matched once. Recall is the share of real requirements a model found. Precision is the share of its rows that were real.

## Results

Macro F1 averages the three held-out contracts. Full per-contract tables are in [results.md](results.md).

| Contestant | Macro F1 |
|---|---|
| Claude, few-shot, no tuning | 0.52 |
| Inkling, LoRA round 2 | 0.53 |
| Qwen3.5-4B, LoRA round 2 | 0.55 |
| Kimi K2.6, LoRA round 2 | 0.57 |
| Jev, sentence selection | 0.64 |

Claude found 27% to 52% of the real requirements, with 52% to 91% precision. Fine-tuning on 11 practice contracts lifted each open model to about Claude's level. The second round changed only the training labels, not the recipe.

Jev works differently. Code splits the RFP into sentences, and Jev decides for each sentence whether it is a requirement and how to label it. A row's text is always a sentence from the document, so a row can be wrong but cannot be invented. Jev found 74% to 88% of the real requirements, with 45% to 60% precision. Its confidence scores did not pass the calibration check: in the top confidence bin, the share of real requirements was 12 points off the stated confidence, and the bar was 10.

`out/jev/splitter_ceiling.json` shows how many gold rows the sentence splitter can reach at all. That number caps Jev's recall.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest evals -q         # scorer and harness tests, no API calls
```

The scored outputs are committed, so you can re-score without calling any model:

```bash
python evals/score.py data/gold/3202600219.json out/baselines/claude-3202600219.json
python scripts/jev_eval.py        # rescores out/jev/*.sentences.json
python scripts/make_results.py    # rebuilds results.md
```

To run the models yourself, put keys in `.env`:

- `ANTHROPIC_API_KEY` for the Claude baseline and the teacher scripts.
- `TYPESAFE_API_KEY` for `src/jev_extract.py`.
- Tinker access for the open models. Install `tinker` and `tinker_cookbook`, then:

```bash
python src/baseline.py claude --gold
python src/train_lora.py alias=kimi
python src/baseline.py kimi --gold
python src/jev_extract.py 3202600219
```

`src/train_lora.py` uses one recipe for every contestant: LoRA rank 32, learning rate 1e-4 with linear decay, batch 4, 4 epochs. `out/checkpoints.json` lists the sampler paths of the tuned runs.

## Layout

| Path | What it holds |
|---|---|
| `src/` | Extraction backends, prompts, teacher labeling, LoRA training |
| `evals/` | The scorer and the tests |
| `scripts/` | Jev scoring, training-data builders, results table |
| `data/` | RFPs, gold keys, teacher labels, training pairs |
| `out/` | Saved predictions and raw model text for every scored run |

The repo covers extraction only. The production layer that matches requirements to a firm's past work is not included.
