# NUMS tool-use fine tuning

NUMS currently uses `qwen2.5:1.5b-instruct` through Ollama. This folder provides a small, synthetic **starter** dataset, a deterministic train/validation/test split, and a read-only tool-call evaluator. The starter examples are for checking the pipeline; they do not establish that a fine-tuned model is better.

See [BASELINE.md](BASELINE.md) for the current starter-set result.

## Collect labels

Add reviewed examples to `seed_examples.jsonl` or a separate private JSONL file. Each record needs `id`, `user`, `tool`, and `arguments`. Labels should be the **first** correct tool call for the user's request. Remove personal paths, secrets, and private content before using actual NUMS conversations. Keep evaluation examples separate from training examples, and include cases where a tool should not be called before deploying a trained model.

Set `NUMS_TRACE_FILE=~/.local/share/nums/usage.nums-trace.jsonl` when running NUMS to record prompts, proposed tool calls, replies, and model errors locally. The trace intentionally omits tool results and is owner-readable only, but prompts and tool arguments can still contain private information. Review and correct traces before converting them into labeled training examples; model proposals are not ground truth.

Prepare a review file, edit each record's expected `tool`, `arguments`, and
optional `answer`, then set `reviewed` to `true` only after checking it. Export
skips every unreviewed record:

```bash
PYTHONPATH=src:. .venv/bin/python training/review_traces.py prepare ~/.local/share/nums/usage.nums-trace.jsonl training/private/review.jsonl
PYTHONPATH=src:. .venv/bin/python training/review_traces.py export training/private/review.jsonl training/private/examples.jsonl
```

## Prepare and evaluate

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python training/build_dataset.py training/seed_examples.jsonl training/data
PYTHONPATH=src .venv/bin/python training/evaluate.py training/data/test.jsonl --model qwen2.5:1.5b-instruct
```

The evaluator sends prompts to Ollama and **does not execute** the model's requested tools. Compare the same held-out set against both models. Do not switch NUMS to a trained model unless it improves tool choice and full argument accuracy on representative held-out tasks.

## Train on Apple silicon

Install `mlx-lm[train]` in a separate Python environment. MLX-LM accepts the tool-call JSONL format generated here and supports LoRA for Qwen2 family models. Use the matching, full-precision `Qwen/Qwen2.5-1.5B-Instruct` base for both training and fusion:

```bash
python -m mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data training/data --mask-prompt --batch-size 1 --num-layers 8 --iters 200 --adapter-path training/adapters
python -m mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --test --data training/data --adapter-path training/adapters
python -m mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path training/adapters --save-path training/fused_model
```

Import the fused Safetensors directory into Ollama with a `Modelfile` containing `FROM /absolute/path/to/training/fused_model`, then run `ollama create nums-tools -f Modelfile`. Check the import with a live evaluation before setting `NUMS_MODEL=nums-tools`. Model weights and generated data stay untracked.

The current seed set is far too small to justify the 200-step recipe as a production model. Replace or expand it with varied, correctly labeled examples and track held-out accuracy before training for use in NUMS.
