# GitHub Repository Summarizer

A lightweight API service that takes a public GitHub repository URL and returns a structured, LLM-generated summary of the project — what it does, which technologies it uses, and how it is organized.

---

## Quick start

### 1. Prerequisites

- Python 3.10 or later
- A Nebius Token Factory API key (or OpenAI / Anthropic key)

### 2. Clone / download the project

```bash
# If you have git
git clone <your-repo-url>
cd repo-summarizer

# Or unzip the archive
unzip repo-summarizer.zip
cd repo-summarizer
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set your API key

```bash
# Nebius Token Factory (preferred)
export NEBIUS_API_KEY="your_nebius_api_key_here"

# — or — OpenAI
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional: raise GitHub API rate limit from 60 to 5 000 requests/hour
export GITHUB_TOKEN="your_github_personal_access_token"
```

### 6. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server is now running at `http://localhost:8000`.

---

## Usage

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

**Example response:**

```json
{
  "summary": "**Requests** is a widely-used Python HTTP library designed to make HTTP/1.1 requests simple and human-friendly...",
  "technologies": ["Python", "urllib3", "certifi", "charset-normalizer", "idna"],
  "structure": "The main library code lives in `src/requests/`. Tests are in `tests/`, documentation source in `docs/`. Package metadata is in `pyproject.toml` and `setup.cfg`."
}
```

**Health check:**

```bash
curl http://localhost:8000/health
```

---

## Model choice

**Model:** `meta-llama/Meta-Llama-3.1-70B-Instruct` via Nebius Token Factory.

This model was chosen because it has a 128 k context window (important for larger repos), produces reliably structured JSON output at `temperature=0.2`, and is available on Nebius with a generous free credit allowance. If an OpenAI key is provided instead, the service falls back to `gpt-4o-mini`.

---

## Repository content strategy

### What we include

| Priority | Files |
|----------|-------|
| Highest | README, `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod` — these immediately reveal purpose and dependencies |
| High | Entry-point files (`main.py`, `app.py`, `index.js`), Dockerfiles, CI configs |
| Normal | Source files, ordered shortest path first (top-level files are usually more representative) |

### What we skip

- **Binary and media files** (`.png`, `.jpg`, `.pdf`, `.whl`, `.exe`, …) — no textual signal
- **Lock files** (`package-lock.json`, `poetry.lock`, `yarn.lock`, …) — machine-generated, enormous, zero human signal
- **Generated / vendored directories** (`node_modules/`, `dist/`, `build/`, `venv/`, `__pycache__/`, …)
- **Data files** (`.csv`, `.parquet`, `.db`) — not relevant to understanding structure

### Context management

1. The full **directory tree** (up to 80 entries) is always included — it is small and gives the LLM a structural overview.
2. Files are fetched in priority order. Each file is **capped at 4 000 characters** (enough to understand purpose, not so much that one file dominates).
3. A **total character budget of 28 000 chars** (~7 000 tokens) is enforced across all file contents. Once the budget is spent, lower-priority files are skipped.
4. This means the LLM always receives the *most informative* subset of the repo, regardless of repo size.

---

## Error handling

| Scenario | HTTP status |
|----------|-------------|
| Invalid or non-GitHub URL | 422 |
| Repository not found / private | 404 |
| GitHub rate limit exceeded | 403 (with tip to set `GITHUB_TOKEN`) |
| Empty repository | 422 |
| GitHub API timeout | 504 |
| LLM returns malformed response | 502 |
| No API key configured | 500 |
