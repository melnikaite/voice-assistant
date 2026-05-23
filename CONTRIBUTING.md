# Contributing

Thanks for looking.  This is a small project run by a small team —
PRs that fix bugs, add tests, improve docs, or expand the tool surface
are welcome.

## Code style

- **Python**: PEP-8, type hints where they help, docstrings on every
  module + non-trivial function.  We don't run black or autoformat —
  match the surrounding style.
- **JavaScript**: vanilla ES2020, no framework, no build step.  Match
  the existing two-space indent + single quotes.
- **Comments**: explain the WHY, not the WHAT.  Reading the code
  tells you what; the comment tells you why this approach over the
  obvious alternative.
- **English** in code, comments, prompts.  User-facing strings go
  through `orchestrator/app/i18n.py`.

## Running tests

The orchestrator has a pytest suite.  It runs in the container so the
Python deps are pinned by the Dockerfile:

```bash
docker exec va-orchestrator pytest /app/tests -q
```

Tests run offline — no LM Studio, mlx-whisper, xtts-server, or
desktop-agent are required.  Adapters are stubbed via `unittest.mock`.
The full suite is ~123 tests, runs in ~10 s.

To iterate on a single test:

```bash
docker exec va-orchestrator pytest /app/tests/test_computer_use.py::test_computer_use_executes_safe_script -v
```

If you change Python dependencies in `orchestrator/pyproject.toml`,
rebuild the image:

```bash
docker compose up -d --build orchestrator
```

## PR workflow

1. Open an issue first for non-trivial changes.  Aligns scope before
   you spend time.
2. Branch from `main`, name it `feature/<short>` or `fix/<short>`.
3. Keep commits focused — one logical change per commit.  Squash
   noisy WIP commits before opening the PR.
4. Commit messages: short imperative subject (≤72 char), blank line,
   body explaining the WHY.  See `git log --oneline` for the style.
5. Run the test suite locally before pushing.
6. Open a PR with: what changed, why, how you tested it, and any
   risks/follow-ups.

## What's in scope

- Bug fixes anywhere
- Test additions for under-covered paths
- New LLM tools (see [`docs/adding-a-tool.md`](docs/adding-a-tool.md))
- New locale (see [`docs/adding-a-locale.md`](docs/adding-a-locale.md))
- New desktop-agent backend (Linux / Windows polish)
- Documentation improvements
- Provider integrations (Ollama / vLLM / cloud LLM behind a flag)

## What's out of scope (without prior discussion)

- Multi-tenant / multi-account architecture.  The project is
  intentionally single-household; multi-tenant adds attack surface
  that isn't worth it for the use case.
- Cloud-by-default features.  The local-first stance is a property,
  not a default we want to relax.
- Heavy framework migrations (React/Vue on the frontend; Django on
  the backend).  The simplicity is part of the design.
- Breaking changes to the read-only stance on `computer_use`.  Four
  layers of defence around destructive actions are deliberate.

## Reporting security issues

Don't open a public issue.  Email the maintainers (see
`CODEOWNERS` once we have one, or use the repo's security advisory
feature on GitHub).

## License

By contributing you agree your changes are licensed under the
repository's [MIT license](LICENSE).
