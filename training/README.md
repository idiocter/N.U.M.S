# NUMS tool-use fine tuning

NUMS currently uses `qwen2.5:1.5b-instruct` through Ollama. This folder provides a synthetic **starter** dataset, a deterministic train/validation/test split, and a read-only tool-call evaluator. The starter examples are for checking the pipeline; they do not establish that a fine-tuned model is better.

The seed set includes varied Mac inspection, waiting, action, coding, and Git
requests so new tools participate in pipeline tests. These labels remain
synthetic and must be supplemented with reviewed real-world traces.

Audit seed coverage whenever runtime tools or labels change:

```bash
PYTHONPATH=src:. .venv/bin/python training/coverage.py \
  training/seed_examples.jsonl --minimum-per-tool 2 --require-complete
```

The command reports per-tool counts and fails when any runtime tool is missing
or below the requested minimum. Complete synthetic coverage checks plumbing and
does not replace representative held-out requests.

See [BASELINE.md](BASELINE.md) for the current starter-set result.

## Collect labels

Add reviewed examples to `seed_examples.jsonl` or a separate private JSONL file. Each record needs `id`, `user`, `tool`, and `arguments`. Labels should be the **first** correct tool call for the user's request. Remove personal paths, secrets, and private content before using actual NUMS conversations. Keep evaluation examples separate from training examples, and include cases where a tool should not be called before deploying a trained model.

Use the same optional `group` value for paraphrases or variants of one underlying
request. The builder keeps every member of a group in one split so related
wording cannot leak from training into validation or test data.

Dataset validation rejects repeated user prompts even when their IDs, spacing,
or capitalization differ. This prevents the same request from appearing in
multiple train and evaluation splits.
The generated `split_manifest.json` preserves each case's assignment when more
reviewed labels are added later, keeping older test cases held out. It also
fingerprints each prompt and label, so changing an assigned case requires a new
ID instead of silently rewriting a held-out example.

Set `NUMS_TRACE_FILE=~/.local/share/nums/usage.nums-trace.jsonl` when running NUMS to record prompts, proposed tool calls, replies, and model errors locally. The trace intentionally omits tool results and is owner-readable only, but prompts and tool arguments can still contain private information. Review and correct traces before converting them into labeled training examples; model proposals are not ground truth.

Prepare a review file, edit each record's expected `tool`, `arguments`, and
optional `answer`, then set `reviewed` to `true` only after checking it. Set
`decision` to `accept` when the proposed tool and arguments are correct, or
`corrected` after fixing them. Use `reject` for a trace that should never enter
the dataset; rejected records remain reviewed and are skipped during export.
An `accept` decision is rejected if the label no
longer matches the proposal. Failed model calls preserve their error and start
with an empty proposed tool, so they cannot be mistaken for valid no-tool
answers. Set the optional `group` during review for related paraphrases; export
preserves it for split isolation. Preparation validates trace structure before
changing the review file. Export skips every unreviewed record:

```bash
PYTHONPATH=src:. .venv/bin/python training/review_traces.py prepare ~/.local/share/nums/usage.nums-trace.jsonl training/private/review.jsonl
PYTHONPATH=src:. .venv/bin/python training/review_traces.py export training/private/review.jsonl training/private/examples.jsonl
```

You can rerun `prepare` as new traces arrive; existing review decisions are
preserved even when a trace log is reordered or rotated. Review and export
files are written with owner-only permissions.
An invalid reviewed label leaves the previous export intact.

## Prepare and evaluate

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python training/build_dataset.py training/seed_examples.jsonl training/data
PYTHONPATH=src .venv/bin/python training/evaluate.py training/data/test.jsonl \
  --model qwen2.5:1.5b-instruct --json-out training/runs/baseline.json
```

Add `--resume` to reuse completed cases from the output report and atomically
checkpoint the private report after every case. Cases with request or response
errors are retried when the command resumes. Resume also rejects prior report
cases that no longer exist in the held-out file, preventing a report for a
different test set from being silently reused. Use `--retries 1` through
`--retries 3` to retry transient Ollama connection failures during the run;
malformed model responses are scored immediately rather than retried.

The JSON report includes overall accuracy, a breakdown by expected tool, and
case-level failures and arguments so regressions can be compared before
installing an adapted model. Reports are written with owner-only permissions
under `training/runs/` by default because arguments can contain private paths.
Each case also carries a fingerprint of its prompt, system instructions, tool
schemas, and expected label. Report comparison rejects mismatched fingerprints.
The evaluator validates the complete held-out file, including unique IDs and
fingerprint format, before it sends any request to Ollama. When a case includes
a fingerprint, the evaluator recomputes it and rejects edits to the prompt,
tools, or expected label made after the split was built.

The evaluator sends prompts to Ollama and **does not execute** the model's requested tools. Compare the same held-out set against both models. Do not switch NUMS to a trained model unless it improves tool choice and full argument accuracy on representative held-out tasks.

After evaluating an adapted model on the same held-out data, compare reports:

```bash
PYTHONPATH=src:. .venv/bin/python training/compare_reports.py \
  training/runs/baseline.json training/runs/candidate.json
```

The comparison rejects changed labels, a lower tool choice score, any per-tool
regression, increased model or connection response errors, or a candidate that
does not improve full tool and argument accuracy. Individual request failures
are scored as incorrect while the remaining held-out cases continue.

Generated split files are replaced atomically with owner-only permissions,
since real reviewed examples can contain private requests and paths.

## Train on Apple silicon

Install `mlx-lm[train]` in a separate Python environment. MLX-LM accepts the tool-call JSONL format generated here and supports LoRA for Qwen2 family models. Use the matching, full-precision `Qwen/Qwen2.5-1.5B-Instruct` base for both training and fusion:

```bash
python -m mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data training/data --mask-prompt --batch-size 1 --num-layers 8 --iters 200 --adapter-path training/adapters
python -m mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --test --data training/data --adapter-path training/adapters
python -m mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path training/adapters --save-path training/fused_model
```

Import the fused Safetensors directory into Ollama with a `Modelfile` containing `FROM /absolute/path/to/training/fused_model`, then run `ollama create nums-tools -f Modelfile`. Check the import with a live evaluation before setting `NUMS_MODEL=nums-tools`. Model weights and generated data stay untracked.

The current seed set is far too small to justify the 200-step recipe as a production model. Replace or expand it with varied, correctly labeled examples and track held-out accuracy before training for use in NUMS.
