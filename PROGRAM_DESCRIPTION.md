Program Description — Model Tuning Agent

1. Purpose

This document explains the architecture and runtime behavior of the Model Tuning Agent. The agent automates sequential, single-parameter tuning rounds guided by a GPT model that suggests parameter changes and summaries.

2. High-level architecture

- `run.py` orchestrates the whole process: load configuration, ask GPT for an initial plan, loop over priority parameters, run inner training rounds, and request GPT suggestions until all parameters are tuned.
- `config.py` holds the default configuration, the list of tunable keys, and agent settings (GPT model, temperature, etc.).
- `core/training.py` exposes training utilities: building a training config, training a model for a single round, evaluating results, and exporting metrics.
- `agents/gpt_agent.py` contains functions that format prompts, call the OpenAI API through the client, and parse GPT responses into structured suggestions.
- `utils/openai_client.py` provides API key management and a thin wrapper around the OpenAI HTTP client.
- `utils/report_generator.py` writes per-run and per-round human-readable reports to `docs/reports/`.

3. Workflow and control flow

1. The program starts in `run.py`. It loads the defaults from `config.py` and merges optional overrides.
2. The agent calls `ask_gpt_for_initial_plan` to request an initial `base_config` and an ordered list of `priority_keys` to tune.
3. For each `priority_key` (up to a configured maximum):
   - Initialize per-parameter defaults and history structures.
   - Start inner rounds (1..R where R is controlled by `ROUNDS_PER_PARAM`).
   - Each inner round calls `train_one_round` with the current configuration subsection.
   - After training, evaluate using STS-B style evaluator or configured metric; record the metric in history.
   - Request GPT for a suggestion specific to the focused key via `ask_gpt_for_new_config`.
   - If GPT suggests a valid change for the focused key, apply it and continue inner rounds until stopping criteria are met.
   - Fix the best-performing value for that parameter into the global configuration.
4. After all priority keys are processed, optionally copy the best model and call `ask_gpt_for_overall_summary` to generate a final human-readable summary.
5. Report generation writes a bilingual (English/Chinese) summary of each round and the final recommendations into `docs/reports/`.

4. Important modules and functions

- `core/training.py`
  - `make_default_config()` — constructs the default configuration dictionary used by the run.
  - `train_one_round(config, output_dir)` — executes a single training round with the provided config, saves artifacts, and returns evaluation metrics.
  - `set_global_seed(seed)` — ensure deterministic behavior for reproducible runs.

- `agents/gpt_agent.py`
  - `ask_gpt_for_initial_plan(context)` — returns `{base_config, priority_keys}`.
  - `ask_gpt_for_new_config(history, current_config, key)` — returns a suggestion limited to the `key` and optional rationale.
  - `ask_gpt_for_overall_summary(history)` — returns a final text summary and suggestions for future runs.

- `utils/openai_client.py`
  - `_load_api_key()` — loads API key using `.env` (preferred) then environment variables fallback.
  - `client.chat(messages, **kwargs)` — wrapper to call the chosen GPT model.

- `utils/report_generator.py`
  - `generate_run_report(run_metadata, rounds_history)` — writes a Markdown report into `docs/reports/` with timestamps and key metrics.

5. Configuration

All user-tunable values live in `config.py`. Typical keys of interest:
- NUM_TRAIN_EPOCHS
- BATCH_SIZE
- LEARNING_RATE
- WARMUP_STEPS
- ROUNDS_PER_PARAM
- MAX_PRIORITY_PARAMS

Tunable keys are documented in `config.py` in a single list `TUNABLE_KEYS`.

6. Logging and artifacts

- Model checkpoints and training artifacts are saved under `models/` (or the configured path in `config.py`).
- Generated run reports are placed under `docs/reports/`.
- The agent maintains an in-memory `history_for_agent` structure per run; persistent logging is written to `docs/reports/` and optional log files.

7. Recoverability and backups

- The project should be run under version control (git). Deleted files can be restored from git history.
- If you remove files from the working tree, preserve a backup zip if you require safety before cleaning.

8. Extensibility

- To add a new training dataset or metric, implement data loading and evaluator code in `core/training.py` and expose a configuration switch in `config.py`.
- To change GPT prompting behavior, modify or extend `agents/gpt_agent.py`.

9. Operational recommendations

- For development and testing, run with reduced dataset sizes (see `QUICK_START.md`).
- Use a virtual environment and isolate package versions.
- Keep `config.py` under version control and use local overrides for machine-specific paths.

10. Troubleshooting

- If GPT calls fail, check `OPENAI_API_KEY` availability and network connectivity.
- If training is slow, reduce `NUM_TRAIN_EPOCHS` or `BATCH_SIZE` for local tests.

11. Next steps to standardize repository

- Optionally remove the old auxiliary files listed in `FILE_CLASSIFICATION.md` after confirming backups or git restore plan.
- Add `requirements.txt` containing exact dependency versions for reproducible installs.
- Add a `LICENSE` and optional `CONTRIBUTING.md` for external collaboration.

