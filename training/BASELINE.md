# Starter evaluation: 2026-09-19

Model: `qwen2.5:1.5b-instruct` in Ollama.

The 40 synthetic examples were split by `build_dataset.py` into 24 train, 8 validation, and 8 test cases. The unchanged model scored **8/8** on tool choice and **8/8** on tool plus arguments on the test split. The evaluator did not execute tools.

This is a pipeline smoke test, not evidence of real-world tool reliability. The examples are simple and share patterns across splits. A useful fine-tuning decision needs representative, independently reviewed NUMS requests, including failures and cases requiring no tool. Keep those examples out of the training split and compare the base and trained models on the same held-out set.
