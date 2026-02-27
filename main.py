"""
GitHub Repository Summarizer API

POST /summarize  — accepts a GitHub URL, fetches the repo contents intelligently,
                   and returns an LLM-generated summary of the project.
GET  /health     — simple liveness check.
"""

import os
import re
import json
import base64
import logging

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from openai import OpenAI

# -- Logging ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# -- App ----------------------------------------------------------------------
app = FastAPI(title="GitHub Repo Summarizer", version="1.0.0")

# -- API keys (injected via .env file) ----------------------------------------
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")  # optional: raises GitHub rate limit 60 → 5000 req/hr


def _get_llm_client() -> tuple[OpenAI, str]:
    """Return (client, model) for whichever API key is present."""
    if NEBIUS_API_KEY:
        return OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=NEBIUS_API_KEY,
        ), "meta-llama/Llama-3.3-70B-Instruct"
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini"
    if GEMINI_API_KEY:
        return OpenAI(api_key=GEMINI_API_KEY), "gemini-2.5-flash"
    raise RuntimeError("No LLM API key found. Set NEBIUS_API_KEY or OPENAI_API_KEY or GEMINI_API_KEY.")


# -- File selection -----------------------------------------------------------

# These files give the LLM the best signal — fetched first, in priority order.
PRIORITY_FILES = [
    "readme.md", "readme.rst", "readme.txt", "readme",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "cargo.toml", "go.mod", "pom.xml",
    "requirements.txt", "pipfile",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile", ".github/workflows",
    "main.py", "app.py", "index.js", "index.ts", "main.go", "main.rs",
]

# Binary, generated, or irrelevant file extensions — skipped entirely.
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",   # images
    ".mp3", ".mp4", ".wav", ".avi", ".mov",                      # media
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",       # archives
    ".exe", ".dll", ".so", ".dylib", ".whl", ".egg",             # binaries
    ".pyc", ".pyo", ".class", ".o", ".a",                        # compiled
    ".csv", ".tsv", ".parquet", ".db", ".sqlite",                # data files
    ".min.js", ".min.css",                                       # minified
    ".woff", ".woff2", ".ttf", ".otf", ".eot",                   # fonts
}

# Directory names that signal vendored, generated, or irrelevant content.
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".pytest_cache",
    "dist", "build", ".next", ".nuxt", "coverage",
    "vendor", "third_party", "venv", ".venv", "env", ".tox",
}

# Lock files are enormous and carry zero human-readable signal.
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "pipfile.lock", "cargo.lock",
    "composer.lock", "gemfile.lock", ".ds_store",
}

# Per-file character cap — enough to understand purpose without one file dominating.
MAX_FILE_CHARS = 4_000
# Total character budget across all fetched files (~7 000 tokens).
CONTEXT_BUDGET_CHARS = 28_000


# -- Pydantic models ----------------------------------------------------------

class SummarizeRequest(BaseModel):
    github_url: str

    @field_validator("github_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not re.match(r"https?://github\.com/[^/]+/[^/]+", v):
            raise ValueError("Must be a valid GitHub repository URL")
        return v


class SummarizeResponse(BaseModel):
    summary: str
    technologies: list[str]
    structure: str


# -- GitHub helpers -----------------------------------------------------------

def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub URL."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1), m.group(2).removesuffix(".git")


def _should_skip(path: str) -> bool:
    """Return True if this file should be excluded from LLM context."""
    lower = path.lower()
    filename = lower.split("/")[-1]
    return (
        filename in SKIP_FILENAMES
        or any(lower.endswith(ext) for ext in SKIP_EXTENSIONS)
        or bool(set(lower.split("/")) & SKIP_DIRS)
    )


def _priority_score(path: str) -> int:
    """Lower score = higher priority. Unprioritised files get the maximum score."""
    lower = path.lower()
    for i, pattern in enumerate(PRIORITY_FILES):
        if lower == pattern or lower.endswith("/" + pattern) or lower.startswith(pattern):
            return i
    return len(PRIORITY_FILES)


def _fetch_repo_contents(owner: str, repo: str, http: httpx.Client) -> dict:
    """
    Fetch repo metadata and selected file contents from the GitHub API.
    Returns: { meta: dict, tree: str, files: dict[path, content] }
    """
    headers = _github_headers()

    # 1. Repo metadata (description, language, stars, default branch)
    log.info("⏳ [1/3] Fetching repo metadata...")
    resp = http.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
    if resp.status_code == 404:
        raise HTTPException(404, "Repository not found or is private")
    if resp.status_code == 403:
        raise HTTPException(403, "GitHub rate limit exceeded — set the GITHUB_TOKEN env var to increase it")
    resp.raise_for_status()
    meta = resp.json()
    log.info("✅ Found: %s (%s stars, language: %s)",
             meta.get("full_name"), meta.get("stargazers_count"), meta.get("language") or "N/A")

    # 2. Full recursive file tree for the default branch
    log.info("⏳ [2/3] Fetching file tree...")
    branch = meta.get("default_branch", "main")
    tree_resp = http.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
        headers=headers,
    )
    if tree_resp.status_code == 409:
        return {"meta": meta, "tree": "", "files": {}}  # empty repo
    tree_resp.raise_for_status()
    all_items = tree_resp.json().get("tree", [])

    # Separate blobs (files) from trees (dirs), filter out noise
    all_blob_paths = [i["path"] for i in all_items if i["type"] == "blob"]
    candidate_files = [i for i in all_items if i["type"] == "blob" and not _should_skip(i["path"])]
    candidate_files.sort(key=lambda f: (_priority_score(f["path"]), len(f["path"])))
    log.info("✅ Tree: %d total files → %d selected for analysis (rest skipped as noise)",
             len(all_blob_paths), len(candidate_files))

    tree_str = _build_tree_string(all_blob_paths)

    # 3. Fetch file contents until the character budget is spent
    log.info("⏳ [3/3] Fetching file contents (budget: %d chars)...", CONTEXT_BUDGET_CHARS)
    fetched_files = {}
    budget = CONTEXT_BUDGET_CHARS
    for item in candidate_files:
        if budget <= 0:
            log.info("   💰 Budget exhausted — stopping early")
            break
        path = item["path"]
        try:
            r = http.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}", headers=headers)
            if r.status_code != 200:
                continue
            data = r.json()
            raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace") \
                  if data.get("encoding") == "base64" else data.get("content", "")

            # Truncate oversized files with a note so the LLM knows it's partial
            if len(raw) > MAX_FILE_CHARS:
                raw = raw[:MAX_FILE_CHARS] + f"\n... [truncated — {len(raw)} chars total]"

            fetched_files[path] = raw
            budget -= len(raw)
            log.info("   📄 %s (%d chars, budget remaining: %d)", path, len(raw), budget)
        except Exception as e:
            log.warning("   ⚠️  Skipping %s: %s", path, e)

    log.info("✅ Fetched %d files, total context: %d chars",
             len(fetched_files), CONTEXT_BUDGET_CHARS - budget)

    return {"meta": meta, "tree": tree_str, "files": fetched_files}


def _build_tree_string(paths: list[str], max_lines: int = 80) -> str:
    """Compact sorted file listing, capped at max_lines."""
    lines = sorted(paths)[:max_lines]
    if len(paths) > max_lines:
        lines.append(f"... and {len(paths) - max_lines} more files")
    return "\n".join(lines)


# -- LLM summarization --------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior software engineer analysing a GitHub repository.

Respond ONLY with a valid JSON object — no markdown fences, no extra text:
{
  "summary": "<one or two paragraphs: what the project does, who it's for, notable features>",
  "technologies": ["<specific tech, e.g. FastAPI not just Python web>", "..."],
  "structure": "<one paragraph: directory layout and role of key files/folders>"
}

Be concise and accurate. Do not invent features not evidenced by the provided files.
"""


def _build_user_message(meta: dict, tree: str, files: dict) -> str:
    """Assemble the prompt context sent to the LLM."""
    parts = [
        f"# Repository: {meta.get('full_name', 'unknown')}",
        f"GitHub description: {meta.get('description') or 'N/A'}",
        f"Primary language: {meta.get('language') or 'N/A'}",
        f"Stars: {meta.get('stargazers_count', 0)}",
        "",
        f"## File tree (first 80 entries)\n```\n{tree}\n```",
        "",
        "## Selected file contents",
    ]
    for path, content in files.items():
        parts.append(f"\n### {path}\n```\n{content}\n```")
    return "\n".join(parts)


def _call_llm(user_message: str) -> SummarizeResponse:
    """Send context to the LLM and parse the structured JSON response."""
    client, model = _get_llm_client()
    log.info("⏳ Sending %d chars to %s...", len(user_message), model)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,  # low temperature = consistent, structured output
        max_tokens=1024,
    )

    log.info("✅ LLM responded — parsing JSON...")
    raw = completion.choices[0].message.content.strip()

    # Strip markdown fences in case the model adds them despite instructions
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("LLM returned non-JSON: %s", raw[:300])
        raise HTTPException(502, f"LLM returned invalid JSON: {e}")

    for field in ("summary", "technologies", "structure"):
        if field not in data:
            raise HTTPException(502, f"LLM response missing field: '{field}'")

    # Normalise technologies to a list in case the model returns a string
    if not isinstance(data["technologies"], list):
        data["technologies"] = [data["technologies"]]

    return SummarizeResponse(**data)


# -- Endpoints ----------------------------------------------------------------

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    """Fetch a GitHub repo and return an LLM-generated summary."""
    try:
        owner, repo = _parse_github_url(request.github_url)
    except ValueError as e:
        raise HTTPException(422, str(e))

    log.info("🚀 Summarising %s/%s", owner, repo)

    try:
        with httpx.Client(timeout=30) as http:
            repo_data = _fetch_repo_contents(owner, repo, http)
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(504, "GitHub API timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"GitHub API error: {e.response.status_code}")
    except Exception as e:
        log.exception("Unexpected error fetching repo")
        raise HTTPException(500, f"Failed to fetch repository: {e}")

    if not repo_data["files"] and not repo_data["tree"]:
        raise HTTPException(422, "Repository appears to be empty")

    try:
        result = _call_llm(_build_user_message(repo_data["meta"], repo_data["tree"], repo_data["files"]))
        log.info("🎉 Done — %d technologies detected", len(result.technologies))
        return result
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        log.exception("Unexpected error calling LLM")
        raise HTTPException(500, f"LLM call failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}