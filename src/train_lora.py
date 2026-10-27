"""LoRA fine-tune of one contestant on the 38 chunk-level teacher pairs.

Same recipe for all three contestants (the fairness rule for this bake-off):
rank 32, LR 1e-4 with linear decay, batch 4, 4 epochs over
data/pairs/train.jsonl. Only the model and its chat renderer differ; the
renderer disables thinking where the family supports it, because emitting the
matrix directly (instead of thinking out loud past the output budget) is the
exact behavior we're training in.

Run in a venv that has tinker and tinker_cookbook installed:
  python src/train_lora.py alias=qwen
  python src/train_lora.py alias=kimi
  python src/train_lora.py alias=inkling

Checkpoints land in logs/<alias>/checkpoints.jsonl; the final sampler path is
also appended to out/checkpoints.json for src/baseline.py's tuned backends.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chz
import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

ROOT = Path(__file__).resolve().parent.parent

CONTESTANTS = {
    "qwen": ("Qwen/Qwen3.5-4B", "qwen3_5_disable_thinking"),
    "kimi": ("moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking"),
    "inkling": ("thinkingmachines/Inkling", "tml_v0"),  # no thinking toggle
}
# Round-2 aliases: same models, same renderers, same recipe — only the
# training data changes (relabeled + miss-mined pairs). Separate alias keeps
# round-1 logs/ and checkpoint registry entries intact for the article.
#   python src/train_lora.py alias=qwen2 \
#       train_file=data/pairs/train_v2.jsonl
for _alias in ("qwen", "kimi", "inkling"):
    CONTESTANTS[_alias + "2"] = CONTESTANTS[_alias]


@chz.chz
class Config:
    alias: str = "qwen"
    base_url: str | None = None
    train_file: str = str(ROOT / "data" / "pairs" / "train.jsonl")
    # The fixed shared recipe. Do not vary per contestant.
    batch_size: int = 4
    n_epochs: int = 4
    learning_rate: float = 1e-4
    max_length: int = 16384  # longest pair is ~13.6K tokens on every tokenizer
    lora_rank: int = 32
    save_every: int = 10
    ttl_seconds: int | None = 604800


def load_env() -> None:
    import os
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main(config: Config) -> None:
    load_env()
    model_name, renderer_name = CONTESTANTS[config.alias]
    if config.alias.endswith("2") and "v2" not in Path(config.train_file).name:
        raise SystemExit(f"round-2 alias {config.alias!r} must train on the v2 "
                         f"pairs — pass train_file=data/pairs/train_v2.jsonl")
    log_path = str(ROOT / "logs" / config.alias)
    ml_logger = ml_log.setup_logging(
        log_dir=log_path, wandb_project=None, wandb_name=None,
        config=config, do_configure_logging_module=True,
    )
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    rows = [json.loads(line)
            for line in Path(config.train_file).read_text().splitlines()
            if line.strip()]
    print(f"{config.alias}: {model_name} renderer={renderer_name} "
          f"{len(rows)} pairs")

    service_client = tinker.ServiceClient(base_url=config.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
        start_step = resume_info.batch
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_name, rank=config.lora_rank
        )
        start_step = 0

    batches_per_epoch = len(rows) // config.batch_size
    total_steps = batches_per_epoch * config.n_epochs

    for step in range(start_step, total_steps):
        epoch, batch_idx = divmod(step, batches_per_epoch)
        if batch_idx == 0:
            random.Random(epoch).shuffle(rows)

        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client, name=f"{step:06d}",
                log_path=log_path, kind="state",
                loop_state={"batch": step}, ttl_seconds=config.ttl_seconds,
            )

        lr = config.learning_rate * max(0.0, 1.0 - step / total_steps)
        adam_params = tinker.AdamParams(
            learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

        batch_rows = rows[batch_idx * config.batch_size:
                          (batch_idx + 1) * config.batch_size]
        batch = [
            conversation_to_datum(
                row["messages"], renderer, config.max_length,
                renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            for row in batch_rows
        ]

        fwd_bwd_future = training_client.forward_backward(
            batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_future.result()
        optim_result = optim_step_future.result()
        metrics = dict(optim_result.metrics or {})
        metrics["epoch"] = epoch
        metrics["lr"] = lr
        ml_logger.log_metrics(metrics=metrics, step=step)

    paths = checkpoint_utils.save_checkpoint(
        training_client=training_client, name="final", log_path=log_path,
        kind="both", loop_state={"batch": total_steps}, ttl_seconds=None,
    )
    ml_logger.close()

    # Record the sampler path where the eval harness looks for it.
    reg_path = ROOT / "out" / "checkpoints.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg[config.alias] = {
        "base_model": model_name,
        "renderer": renderer_name,
        "sampler_path": paths["sampler_path"],
        "steps": total_steps,
    }
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"done: {total_steps} steps "
          f"({config.n_epochs} epochs x {batches_per_epoch} batches)")
    print(f"sampler checkpoint recorded in {reg_path.relative_to(ROOT)}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
