# 🔍 GitHub Repository Summarizer

Paste in any public GitHub URL and get back a clean, structured summary — what the project does, what technologies it uses, and how it's organized. Powered by an LLM via the Nebius Token Factory API.

---

## Setup

### 1. Prerequisites

- Python 3.10 or later
- A **Nebius Token Factory** API key — [sign up here](https://tokenfactory.nebius.com/) (you get $1 free credit, no top-up needed for this project)
- Alternatively, an OpenAI or Gemini API key works too

### 2. Install dependencies

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Create a `.env` file

Create a file called `.env` in the project root and add your API key:

```
NEBIUS_API_KEY=your_key_here
```

Other supported keys (only one is needed):

```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

> **Optional:** Add a `GITHUB_TOKEN` to raise the GitHub API rate limit from 60 to 5,000 requests/hour. Useful if you're testing many repos quickly.
> ```
> GITHUB_TOKEN=your_github_token_here
> ```

### 4. Start the server

```bash
uvicorn main:app --port 8000 --env-file .env --reload
```

The server is running at `http://localhost:8000`. The `--reload` flag automatically restarts it when you edit `main.py`.

---

## Usage

Send a POST request with any public GitHub URL:

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

**On Windows (PowerShell):**

```powershell
(Invoke-WebRequest -Uri "http://localhost:8000/summarize" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"github_url": "https://github.com/psf/requests"}').Content | ConvertFrom-Json | ConvertTo-Json
```

**Example response:**

```json
{
  "summary": "Requests is a widely-used Python HTTP library designed to make HTTP/1.1 requests simple and human-friendly. It abstracts away the complexity of making requests behind a simple API, supporting connection keep-alive, session cookies, SSL verification, and more.",
  "technologies": ["Python", "urllib3", "certifi", "charset-normalizer", "idna"],
  "structure": "The main library code lives in src/requests/. Tests are in tests/, documentation in docs/. Package metadata is defined in pyproject.toml and setup.cfg."
}
```

**Health check:**

```bash
curl http://localhost:8000/health
```

---

## Watching progress

When a request is running, switch to the **uvicorn terminal tab** to see live progress:

```
🚀 Summarising psf/requests
⏳ [1/3] Fetching repo metadata...
✅ Found: psf/requests (52847 stars, language: Python)
⏳ [2/3] Fetching file tree...
✅ Tree: 143 total files → 38 selected for analysis (rest skipped as noise)
⏳ [3/3] Fetching file contents (budget: 28000 chars)...
   📄 README.md (3821 chars, budget remaining: 24179)
   📄 pyproject.toml (1204 chars, budget remaining: 22975)
⏳ Sending 18432 chars to meta-llama/Llama-3.3-70B-Instruct...
✅ LLM responded — parsing JSON...
🎉 Done — 5 technologies detected
```

---

## Model choice

**Model:** `meta-llama/Llama-3.3-70B-Instruct` via Nebius Token Factory.

Chosen for its 128k context window (handles large repos well), reliable structured JSON output, and availability on Nebius with free credits. The service automatically falls back to `gpt-4o-mini` if an OpenAI key is provided instead, or `gemini-2.5-flash` for a Gemini key.

---

## How repository content is selected

Sending an entire repo to an LLM isn't practical — repos can have thousands of files. Instead, the service is selective:

**What gets included (in priority order):**

| Priority | Examples |
|----------|---------|
| Highest | `README`, `pyproject.toml`, `package.json`, `go.mod` — immediately reveal purpose and dependencies |
| High | Entry points (`main.py`, `app.py`, `index.js`), Dockerfiles, CI configs |
| Normal | Other source files, preferring shorter paths (top-level files are most representative) |

**What gets skipped:**

- **Binary and media files** (`.png`, `.pdf`, `.exe`, …) — no text content
- **Lock files** (`package-lock.json`, `poetry.lock`, …) — machine-generated, enormous, zero signal
- **Generated directories** (`node_modules/`, `dist/`, `venv/`, `__pycache__/`, …)
- **Data files** (`.csv`, `.db`, `.parquet`) — irrelevant to code structure

**Context limits:**

- Each file is capped at **4,000 characters** so no single file dominates the context
- A total budget of **28,000 characters** (~7,000 tokens) is enforced across all files
- The directory tree (up to 80 entries) is always included — it's small but gives the LLM the full structural picture

---

## Error reference

| Scenario | HTTP status |
|----------|-------------|
| Invalid or non-GitHub URL | 422 |
| Repository not found or private | 404 |
| GitHub rate limit exceeded | 403 |
| Empty repository | 422 |
| GitHub API timeout | 504 |
| LLM returns malformed response | 502 |
| No API key configured | 500 |