# Contributing to simple_api_router

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Setting Up the Dev Environment

```bash
git clone https://github.com/chen-squared/simple_api_router.git
cd simple_api_router
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are pure unit tests with no network access. You can run them without installing — just `cd` into the project directory and pytest will find the package from the current directory. With the venv:

```bash
/path/to/venv/bin/python -m pytest tests/ -v
```

- Anthropic → OpenAI request conversion (text, tools, vision, cache control, thinking, reasoning effort)
- OpenAI → Anthropic response conversion (text, tool calls, reasoning content)
- Streaming conversion (text, tool calls, thinking, empty deltas)
- OpenAI Responses API conversion (request, response, streaming)
- DeepSeek `reasoning_content` passthrough (request, response, streaming, round-trip)
- Config validation (`api_format`, `deepseek_reasoning`, `model_map`)
- Ported tests from [cc-switch-cli](https://github.com/cc-switch-cli/cc-switch-cli)

To run a single test class:

```bash
python -m pytest tests/test_converter.py::TestDeepSeekReasoning -v
```

## Code Style

No linter is enforced, but please follow the existing conventions:

- Type-annotate function signatures.
- Keep `converter.py` stateless — pure functions and async generators only (no I/O, no config).
- Keep `proxy.py` as the only place that touches HTTP and config.
- Write a test for every new conversion case.
- Keep lines under ~100 characters.

## Module Structure

```
simple_api_router/
  config.py     — Pydantic config models + YAML loader
  app.py        — FastAPI application factory
  proxy.py      — Routing, provider resolution, HTTP dispatch
  converter.py  — Stateless Anthropic ↔ OpenAI format conversion
  logger.py     — Logging setup
  cli.py        — CLI entry point
```

## Submitting a Pull Request

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes and add tests.
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Push your branch and open a PR against `main`.

## Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>
```

Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.

Examples:
- `feat: add per-provider timeout configuration`
- `fix: empty content delta opening spurious text block`
- `docs: add DeepSeek reasoning example to README`

## Questions?

Open a [GitHub Issue](https://github.com/chen-squared/simple_api_router/issues) or start a [Discussion](https://github.com/chen-squared/simple_api_router/discussions).
