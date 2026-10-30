# Contributing

Thanks for taking a look. Small, focused contributions are very welcome: bug fixes, tests, docs, and
anything labelled [good first issue](https://github.com/nisaral/uncertainty-diffusion-world-models/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or [help wanted](https://github.com/nisaral/uncertainty-diffusion-world-models/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

## Getting set up
`pip install -r requirements.txt` (CPU is enough), then `python -m udwm.scripts.smoke_test`.

## Making a change
1. Fork the repo and create a branch.
2. Keep the change small and explain *why* in the PR description.
3. Before opening the PR: `python -m pytest tests -q` passes and the smoke test runs. Any new number must come from a committed run (see REPRODUCE.md).
4. Never commit API keys, `.env` files, or large data/model files.

## Reporting bugs and ideas
Open an issue using the templates. For bugs, include the exact command, the full error and your environment.
If you are unsure whether something is in scope, open an issue first and ask.

By contributing you agree that your work is released under this repository's license.
