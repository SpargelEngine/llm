# Experiment Configs

This directory contains git-managed JSON configs for
`scripts/tianjiao/run_exp.py`. Runtime artifacts are written to `runs/`, which is
ignored by git.

Start a new run:

```bash
uv run python scripts/tianjiao/run_exp.py run exps/example.json
```

Resume an explicit run:

```bash
uv run python scripts/tianjiao/run_exp.py resume runs/example-20260630-120000-abcdef12
```

Fresh runs require:

- the config path is under `exps/`
- the config is tracked by git
- the working tree is clean
- the current branch has no unpushed commits

The runner refuses to overwrite an existing `runs/<tag>` directory. Resume is
only performed by the `resume` subcommand.

## JSON Schema

```json
{
  "schema_version": 1,
  "model": {
    "vocab_size": 8192,
    "max_seq_len": 4096,
    "num_layer": 4,
    "num_head": 4,
    "dim": 256,
    "dim_key": 64,
    "dim_value": 64,
    "dim_ff_hidden": 1024,
    "use_rope": true,
    "ff_activation": "relu"
  },
  "tokenizer": "data/tokenizer.json",
  "data": {
    "train_path": "data/tokens.parquet",
    "validation_path": "data/tokens_val.parquet",
    "start_index": 0,
    "start_offset": 0,
    "add_sot": false,
    "add_eot": true
  },
  "train": {
    "steps": 1000,
    "seq_len": 4096,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.1,
    "micro_batches": 4,
    "log_period": 10,
    "checkpoint_interval": 100,
    "validation_batches": 1,
    "loop_dataset": false,
    "seed": 1234,
    "use_bf16": true,
    "float32_precision": "high"
  }
}
```

`validation_path` and `validation_batches` are optional. If
`validation_batches` is omitted, the runner uses `max(log_period // 10, 1)`.
Relative dataset and tokenizer paths are interpreted relative to the repo root.
The tokenizer path is recorded for reproducibility; training consumes the
pre-tokenized Parquet files directly.
