# Results

Requirement-level scores against the three hand-labeled answer keys
(match = same section + enough word overlap; scorer in evals/score.py).
Recall is the headline number: it is the share of real requirements
the model found. Precision is the share of its rows that were real.

## Baselines (no fine-tuning)

| Model | RFP | found / real | recall | precision | F1 | labels right |
|---|---|---|---|---|---|---|
| Claude (few-shot, the bar to beat) | 3202600219 | 24 / 54 | 44% | 71% | 0.55 | 58% |
| Claude (few-shot, the bar to beat) | HHS0015558v2 | 95 / 181 | 52% | 91% | 0.67 | 76% |
| Claude (few-shot, the bar to beat) | 3202600177 | 54 / 203 | 27% | 52% | 0.35 | 72% |
| Qwen3.5-4B (un-tuned) | 3202600219 | 16 / 54 | 30% | 43% | 0.35 | 62% |
| Qwen3.5-4B (un-tuned) | HHS0015558v2 | 82 / 181 | 45% | 44% | 0.45 | 66% |
| Qwen3.5-4B (un-tuned) | 3202600177 | 0 / 203 | 0% | 0% | 0.00 | n/a |
| Kimi K2.6 (un-tuned) | 3202600219 | 10 / 54 | 18% | 50% | 0.27 | 70% |
| Kimi K2.6 (un-tuned) | HHS0015558v2 | 13 / 181 | 7% | 65% | 0.13 | 69% |
| Kimi K2.6 (un-tuned) | 3202600177 | 34 / 203 | 17% | 41% | 0.24 | 59% |
| Inkling (un-tuned) | 3202600219 | 37 / 54 | 68% | 37% | 0.48 | 65% |
| Inkling (un-tuned) | HHS0015558v2 | 40 / 181 | 22% | 52% | 0.31 | 62% |
| Inkling (un-tuned) | 3202600177 | 39 / 203 | 19% | 26% | 0.22 | 92% |

## After fine-tuning (same scorer, same answer keys)

Each contestant was trained once, with one shared recipe, on the
38 chunk-size pieces of the 11 practice contracts. The three
contracts below stayed locked away during training.

| Model | RFP | found / real | recall | precision | F1 | labels right |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (fine-tuned) | 3202600219 | 16 / 54 | 30% | 73% | 0.42 | 50% |
| Qwen3.5-4B (fine-tuned) | HHS0015558v2 | 92 / 181 | 51% | 64% | 0.57 | 47% |
| Qwen3.5-4B (fine-tuned) | 3202600177 | 87 / 203 | 43% | 45% | 0.44 | 75% |
| Kimi K2.6 (fine-tuned) | 3202600219 | 20 / 54 | 37% | 69% | 0.48 | 50% |
| Kimi K2.6 (fine-tuned) | HHS0015558v2 | 77 / 181 | 42% | 71% | 0.53 | 69% |
| Kimi K2.6 (fine-tuned) | 3202600177 | 75 / 203 | 37% | 35% | 0.36 | 72% |
| Inkling (fine-tuned) | 3202600219 | 25 / 54 | 46% | 81% | 0.59 | 56% |
| Inkling (fine-tuned) | HHS0015558v2 | 107 / 181 | 59% | 14% | 0.22 | 71% |
| Inkling (fine-tuned) | 3202600177 | 87 / 203 | 43% | 42% | 0.42 | 80% |

## Round 2: relabeled training data

For round 2 nothing changed about the training recipe. Only the
training data changed. The practice answers were re-graded against the
labeling rules learned from the hand-checked contracts, and a
second pass hunted down requirement sentences the first teacher
run had missed (586 new rows, growing the training set by half).
The same three locked-away contracts score the result.

| Model | RFP | found / real | recall | precision | F1 | labels right |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (fine-tuned, round 2) | 3202600219 | 25 / 54 | 46% | 71% | 0.56 | 68% |
| Qwen3.5-4B (fine-tuned, round 2) | HHS0015558v2 | 85 / 181 | 47% | 79% | 0.59 | 65% |
| Qwen3.5-4B (fine-tuned, round 2) | 3202600177 | 104 / 203 | 51% | 47% | 0.49 | 76% |
| Kimi K2.6 (fine-tuned, round 2) | 3202600219 | 27 / 54 | 50% | 68% | 0.57 | 56% |
| Kimi K2.6 (fine-tuned, round 2) | HHS0015558v2 | 94 / 181 | 52% | 72% | 0.60 | 67% |
| Kimi K2.6 (fine-tuned, round 2) | 3202600177 | 107 / 203 | 53% | 56% | 0.54 | 77% |
| Inkling (fine-tuned, round 2) | 3202600219 | 24 / 54 | 44% | 55% | 0.49 | 50% |
| Inkling (fine-tuned, round 2) | HHS0015558v2 | 132 / 181 | 73% | 56% | 0.63 | 72% |
| Inkling (fine-tuned, round 2) | 3202600177 | 97 / 203 | 48% | 44% | 0.46 | 81% |

Round-2 notes:

- Averaged across the three contracts, every round-2 model now
  edges Claude on F1 (Claude 0.52; Qwen 0.55; Kimi 0.57;
  Inkling 0.53). Kimi is the round's winner: it beats Claude
  outright on two contracts and ties its recall on the third.
- Inkling's round-1 disaster on HHS0015558v2 (779 rows, 14%
  precision) is fixed: 236 rows, 56% precision, and 73% recall,
  the highest recall any model posted on any contract.
- Claude keeps one crown: precision. 91% on HHS0015558v2 is
  still far above any tuned model, and it wins that contract
  on F1 (0.67).
- The retrained Qwen picked up a repetition habit: on 5 of 9
  chunk runs it started echoing the same row hundreds of times.
  A new harness guard detects this and resamples at higher
  temperature (same guard for every contestant); all 5 runs
  recovered. No such guard existed in round 1, where the same
  disease sank Inkling untreated.

## What the failures looked like (from the saved raw outputs)

- The un-tuned open models' worst enemy was their own habit of
  thinking out loud. On failed runs they filled 100,000+ characters
  with prose notes ('Page 15: ... -> Deadline. Skip.') and ran out
  of room before ever writing the JSON answer. Qwen3.5-4B's zero on
  3202600177 and most Kimi chunks died this way.
- Kimi K2.6 serves with a 32,768-token window on Tinker, so it reads
  each contract in pieces; Qwen does too (small model). Inkling
  (65,536) reads whole contracts but its two big answers were cut
  off mid-list; only the complete rows before the cut were kept.
- Claude's few-shot numbers are the bar: high precision, but it
  finds only a quarter to half of the real requirements. The
  fine-tuning round has to beat that recall to justify itself.

