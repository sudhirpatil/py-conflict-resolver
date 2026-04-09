# py-conflict-resolver

Automatically resolve Python package version conflicts using an LLM-powered agent. Provide a `requirements.txt`, and the tool installs packages in a temporary virtual environment, asks an LLM to fix any conflicts, and retries until it succeeds — then writes a clean resolved file.

---

## Installation

```bash
pip install -e .
```

---

## Setup

1. Copy `.env.example` to `.env` and add your API key for the provider you want to use:

```bash
cp .env.example .env
```

```ini
# .env
OPENAI_API_KEY=sk-...        # if using OpenAI (default)
ANTHROPIC_API_KEY=sk-ant-... # if using Anthropic
GOOGLE_API_KEY=AIza...       # if using Gemini
```

2. (Optional) Edit `config.toml` to change the provider, model, or max loops:

```toml
[llm]
provider = "openai"   # "openai" | "anthropic" | "gemini"
model    = "gpt-4.1"

[agent]
max_loops = 10
```

---

## Usage

**Basic** — resolve conflicts and write `requirements.resolved.txt` next to the input file:

```bash
conflict-resolver requirements.txt
```

**Specify output file:**

```bash
conflict-resolver requirements.txt --output fixed-requirements.txt
```

**Verbose output** (shows each install attempt and LLM interaction):

```bash
conflict-resolver requirements.txt --verbose
```

**Override provider/model on the fly:**

```bash
conflict-resolver requirements.txt --provider anthropic --model claude-sonnet-4-6
conflict-resolver requirements.txt --provider gemini --model gemini-2.5-flash
```

**Limit retry attempts:**

```bash
conflict-resolver requirements.txt --max-loops 5
```

**Use a custom config file:**

```bash
conflict-resolver requirements.txt --config ./my-config.toml
```

**Use a specific .env file:**

```bash
conflict-resolver requirements.txt --env-file /path/to/.env
```

---

## How it works

```
[install packages] → success → write resolved requirements.txt
        |
        | failure
        ↓
[LLM analyzes pip error] → proposes fixed requirements.txt
        |
        ↓
[retry install] → ... (up to max_loops times)
        |
        | still failing after max_loops
        ↓
      exit 1
```

The temporary virtual environment is always deleted on exit, whether the run succeeds, fails, or is interrupted.

---

## Supported providers & models

| Provider  | Models                                                        | Env var           |
|-----------|---------------------------------------------------------------|-------------------|
| OpenAI    | gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, o3, o4-mini     | `OPENAI_API_KEY`  |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001 | `ANTHROPIC_API_KEY` |
| Gemini    | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash           | `GOOGLE_API_KEY`  |

The full list is in `config.toml`. Add a model there before selecting it.
