"""
GitHub Repository Summarizer API
Fetches a public GitHub repo, selects the most informative files,
and returns an LLM-generated summary.
"""

import os
import re
import base64
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="GitHub Repo Summarizer", version="1.0.0")

# ---------------------------------------------------------------------------
# Config — read from environment
# ---------------------------------------------------------------------------
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional, raises rate limit from 60 to 5000 req/hr

# Decide which provider to use
def _get_llm_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) for whichever API key is configured."""
    if NEBIUS_API_KEY:
        client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=NEBIUS_API_KEY,
        )
        # Nebius Token Factory — capable open model with large context
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        return client, model
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client, "gpt-4o-mini"
    # Anthropic uses its own SDK but also has an OpenAI-compatible endpoint
    raise RuntimeError(
        "No LLM API key found. Set NEBIUS_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
    )


# ---------------------------------------------------------------------------
# File-selection constants
# ---------------------------------------------------------------------------

# Files we always try to include (in priority order)
PRIORITY_FILES = [
    "readme.md", "readme.rst", "readme.txt", "readme",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "cargo.toml", "go.mod", "pom.xml",
    "requirements.txt", "pipfile",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile", ".github/workflows",
    "main.py", "app.py", "index.js", "index.ts", "main.go", "main.rs",
    "src/main.py", "src/app.py",
]

# Extensions we skip entirely (binary / generated / irrelevant)
SKIP_EXTENSIONS = {
    # binaries & media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".whl", ".egg",
    ".pyc", ".pyo", ".class", ".o", ".a",
    # data / generated
    ".csv", ".tsv", ".parquet", ".db", ".sqlite",
    ".min.js", ".min.css",
    # fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}

# Path segments that indicate generated / vendored / irrelevant content
SKIP_PATH_SEGMENTS = {
    "node_modules", ".git", "__pycache__", ".pytest_cache",
    "dist", "build", ".next", ".nuxt", "coverage",
    "vendor", "third_party", "venv", ".venv", "env",
    ".tox", "eggs", "*.egg-info",
    "migrations",  # usually auto-generated DB migrations
}

# Lock files — skip them (very large, no useful signal)
SKIP_EXACT_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "pipfile.lock", "cargo.lock",
    "composer.lock", "gemfile.lock",
    ".ds_store", "thumbs.db",
}

# Hard limit on characters we read from a single file
MAX_FILE_CHARS = 4_000
# Approximate token budget for all file contents combined (1 token ≈ 4 chars)
CONTEXT_BUDGET_CHARS = 28_000  # ~7 000 tokens for content


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
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


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1), m.group(2).removesuffix(".git")


def _should_skip(path: str) -> bool:
    """Return True if we should skip this file."""
    lower = path.lower()
    # Check exact file name
    filename = lower.split("/")[-1]
    if filename in SKIP_EXACT_NAMES:
        return True
    # Check extension
    for ext in SKIP_EXTENSIONS:
        if lower.endswith(ext):
            return True
    # Check path segments
    parts = set(lower.split("/"))
    if parts & SKIP_PATH_SEGMENTS:
        return True
    return False


def _priority_score(path: str) -> int:
    """Lower = higher priority."""
    lower = path.lower()
    for i, pattern in enumerate(PRIORITY_FILES):
        if lower == pattern or lower.endswith("/" + pattern) or lower.startswith(pattern):
            return i
    return len(PRIORITY_FILES)


def _fetch_repo_contents(owner: str, repo: str, client: httpx.Client) -> dict:
    """
    Fetch repo metadata and a filtered, prioritised list of file contents.
    Returns a dict with keys: meta, tree, files.
    """
    headers = _github_headers()

    # 1. Repo metadata
    resp = client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Repository not found or is private")
    if resp.status_code == 403:
        raise HTTPException(status_code=403, detail="GitHub rate limit exceeded. Set GITHUB_TOKEN env var.")
    resp.raise_for_status()
    meta = resp.json()

    # 2. Default branch tree (recursive)
    default_branch = meta.get("default_branch", "main")
    tree_resp = client.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1",
        headers=headers,
    )
    if tree_resp.status_code == 409:
        # Empty repo
        return {"meta": meta, "tree": [], "files": {}}
    tree_resp.raise_for_status()
    tree_data = tree_resp.json()

    # 3. Filter blobs (files only, skip dirs and skip-listed paths)
    all_files = [
        item for item in tree_data.get("tree", [])
        if item["type"] == "blob" and not _should_skip(item["path"])
    ]

    # Sort by priority score, then by path length (prefer shorter = top-level)
    all_files.sort(key=lambda f: (_priority_score(f["path"]), len(f["path"])))

    # 4. Build directory tree string (always include, very cheap)
    all_paths = [item["path"] for item in tree_data.get("tree", []) if item["type"] == "blob"]
    tree_str = _build_tree_string(all_paths)

    # 5. Fetch file contents within budget
    budget_remaining = CONTEXT_BUDGET_CHARS
    fetched_files = {}

    for file_item in all_files:
        if budget_remaining <= 0:
            break
        path = file_item["path"]
        try:
            content_resp = client.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                headers=headers,
            )
            if content_resp.status_code != 200:
                continue
            data = content_resp.json()
            if data.get("encoding") == "base64":
                raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            else:
                raw = data.get("content", "")

            # Truncate long files
            if len(raw) > MAX_FILE_CHARS:
                raw = raw[:MAX_FILE_CHARS] + f"\n... [truncated — file is {len(raw)} chars total]"

            fetched_files[path] = raw
            budget_remaining -= len(raw)
        except Exception as e:
            log.warning("Could not fetch %s: %s", path, e)

    return {"meta": meta, "tree": tree_str, "files": fetched_files}


def _build_tree_string(paths: list[str], max_lines: int = 80) -> str:
    """Build a compact directory tree from a list of file paths."""
    lines = sorted(paths)[:max_lines]
    if len(paths) > max_lines:
        lines.append(f"... and {len(paths) - max_lines} more files")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior software engineer. Your task is to analyze a GitHub repository
and produce a structured summary for a developer audience.

Respond ONLY with a valid JSON object (no markdown fences, no extra text) with this shape:
{
  "summary": "<one or two paragraphs describing what the project does and its purpose>",
  "technologies": ["<tech1>", "<tech2>", "..."],
  "structure": "<one paragraph describing how the project is organized>"
}

Guidelines:
- summary: explain what the project does, who it's for, and any notable features or design goals.
- technologies: list programming languages, frameworks, libraries, and key tools. Be specific (e.g. "FastAPI" not just "Python web").
- structure: describe the directory layout and the role of key files/directories.
- Be concise and accurate. Do not hallucinate features not evidenced by the provided files.
"""


def _build_user_message(meta: dict, tree: str, files: dict) -> str:
    parts = [
        f"# Repository: {meta.get('full_name', 'unknown')}",
        f"Description (from GitHub): {meta.get('description') or 'N/A'}",
        f"Primary language: {meta.get('language') or 'N/A'}",
        f"Stars: {meta.get('stargazers_count', 0)}",
        "",
        "## Directory tree (first 80 entries)",
        "```",
        tree,
        "```",
        "",
        "## Selected file contents",
    ]

    for path, content in files.items():
        parts.append(f"\n### {path}\n```\n{content}\n```")

    return "\n".join(parts)


def _call_llm(user_message: str) -> SummarizeResponse:
    """Call the configured LLM and parse the structured response."""
    import json

    client, model = _get_llm_client()
    log.info("Calling LLM model: %s", model)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    raw = completion.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps them anyway
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("LLM returned non-JSON: %s", raw[:500])
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    # Validate required fields
    for field in ("summary", "technologies", "structure"):
        if field not in data:
            raise HTTPException(status_code=502, detail=f"LLM response missing field: {field}")

    if not isinstance(data["technologies"], list):
        data["technologies"] = [data["technologies"]]

    return SummarizeResponse(**data)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    """
    Accepts a GitHub repository URL and returns an LLM-generated summary.
    """
    try:
        owner, repo = _parse_github_url(request.github_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    log.info("Summarizing %s/%s", owner, repo)

    try:
        with httpx.Client(timeout=30) as http:
            repo_data = _fetch_repo_contents(owner, repo, http)
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GitHub API timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e.response.status_code}")
    except Exception as e:
        log.exception("Unexpected error fetching repo")
        raise HTTPException(status_code=500, detail=f"Failed to fetch repository: {e}")

    if not repo_data["files"] and not repo_data["tree"]:
        raise HTTPException(status_code=422, detail="Repository appears to be empty")

    user_message = _build_user_message(
        repo_data["meta"], repo_data["tree"], repo_data["files"]
    )

    try:
        return _call_llm(user_message)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error calling LLM")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
