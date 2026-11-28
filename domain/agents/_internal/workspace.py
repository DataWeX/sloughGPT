"""Agent workspace — build artifacts on disk, path-guarded.

One implementation, two projections: ``ToolRunner`` (the agents page) and
``ToolRegistry`` (chat) both call these functions, so the build capability is
declared once instead of being re-approximated per surface.

Every artifact lives under ``data/agent-projects/<slug>/`` — a dedicated
root, so a model-supplied name can never escape into the checkout. Build
names are slugified (friendly), and the resolved path must stay under the
root (strict); ``write_file`` validates the project name as given, because it
must name an existing project.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

from domain.shared import find_repo_root

PROJECTS_ROOT = os.path.join(find_repo_root(__file__), "data", "agent-projects")

TEMPLATES = ("static", "vite-react", "fastapi")

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class WorkspaceError(ValueError):
    """A request the workspace refuses (bad name, escape attempt, missing project)."""


def slugify(text: str) -> str:
    """Turn free text into a valid project slug (never empty)."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:48].strip("-")
    return slug or "site"


def titleize(text: str) -> str:
    """A human title from a request: drop the build verb, title-case the rest."""
    cleaned = re.sub(
        r"^\s*(?:please\s+)?(?:build|create|make|scaffold|generate)\s+(?:me\s+)?(?:a|an|the)?\s*",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" .!?") or (text or "").strip() or "Site"
    return cleaned[:1].upper() + cleaned[1:]


def _project_dir(name: str) -> str:
    """Resolve and validate a project directory (raises ``WorkspaceError``)."""
    if not _SLUG_RE.match(name or ""):
        raise WorkspaceError(
            f"invalid project name '{name}' — use lowercase letters, digits and dashes"
        )
    root = os.path.realpath(PROJECTS_ROOT)
    path = os.path.realpath(os.path.join(root, name))
    if path != root and not path.startswith(root + os.sep):
        raise WorkspaceError("project path escapes the workspace root")
    return path


def _unique_project(name: str) -> tuple[str, str]:
    """A project name/dir that does not overwrite an existing build."""
    path = _project_dir(name)
    if not os.path.isdir(path) or not os.listdir(path):
        return name, path
    for i in range(2, 100):
        candidate = f"{name}-{i}"
        path = _project_dir(candidate)
        if not os.path.isdir(path) or not os.listdir(path):
            return candidate, path
    raise WorkspaceError(f"too many projects named '{name}' — pick a different name")


def _manifest(root: str) -> list[dict[str, Any]]:
    """Relative path + byte size for every file under ``root``."""
    entries: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            entries.append(
                {
                    "path": os.path.relpath(full, root).replace(os.sep, "/"),
                    "bytes": os.path.getsize(full),
                }
            )
    return sorted(entries, key=lambda e: e["path"])


def _sections(sections: str) -> list[str]:
    names = [s.strip() for s in re.split(r"[,\n]", sections or "") if s.strip()]
    return names[:12] or ["Overview", "Features", "Contact"]


def _static_files(title: str, tagline: str, sections: list[str]) -> dict[str, str]:
    esc = html.escape
    blocks = "\n".join(
        f'      <section id="{slugify(name)}">\n'
        f"        <h2>{esc(name)}</h2>\n"
        f"        <p>Add your {esc(name.lower())} content here.</p>\n"
        "      </section>"
        for name in sections
    )
    index = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(title)}</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <header class="site-header">
      <h1>{esc(title)}</h1>
      <p class="tagline">{esc(tagline)}</p>
    </header>
    <main>
{blocks}
    </main>
    <footer>
      <p>Built with sloughGPT.</p>
    </footer>
    <script src="script.js"></script>
  </body>
</html>
"""
    styles = """/* Base styles — replace freely. */
:root {
  color-scheme: dark;
  --bg: #0f0d16;
  --surface: #171426;
  --text: #ece9f5;
  --muted: #a29bbd;
  --accent: #a78bfa;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}

.site-header {
  padding: 4rem 1.5rem 3rem;
  text-align: center;
  background: linear-gradient(180deg, var(--surface), var(--bg));
}

.site-header h1 { margin: 0 0 0.5rem; font-size: clamp(2rem, 5vw, 3.5rem); }
.tagline { margin: 0; color: var(--muted); }

main {
  max-width: 52rem;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}

section {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: var(--surface);
  border: 1px solid rgba(167, 139, 250, 0.18);
  border-radius: 0.75rem;
}

section h2 { margin-top: 0; color: var(--accent); }

footer {
  padding: 2rem;
  text-align: center;
  color: var(--muted);
  border-top: 1px solid rgba(167, 139, 250, 0.12);
}
"""
    script = """// Progressive enhancement only — the page works without this file.
const sections = document.querySelectorAll("section");

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.style.opacity = "1";
          entry.target.style.transform = "none";
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.15 },
  );

  for (const section of sections) {
    section.style.opacity = "0";
    section.style.transform = "translateY(12px)";
    section.style.transition = "opacity 300ms ease, transform 300ms ease";
    observer.observe(section);
  }
}
"""
    return {"index.html": index, "styles.css": styles, "script.js": script}


def _vite_react_files(name: str, title: str, tagline: str, sections: list[str]) -> dict[str, str]:
    package = f"""{{
  "name": "{name}",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }},
  "dependencies": {{
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  }},
  "devDependencies": {{
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }}
}}
"""
    index = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
"""
    main = """import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
"""
    section_jsx = "\n".join(
        f'      <section id="{slugify(s)}">\n'
        f"        <h2>{html.escape(s)}</h2>\n"
        f"        <p>Add your {html.escape(s.lower())} content here.</p>\n"
        "      </section>"
        for s in sections
    )
    app = f"""export default function App() {{
  return (
    <>
      <header className="site-header">
        <h1>{html.escape(title)}</h1>
        <p className="tagline">{html.escape(tagline)}</p>
      </header>
      <main>
{section_jsx}
      </main>
      <footer>
        <p>Built with sloughGPT.</p>
      </footer>
    </>
  );
}}
"""
    css = """/* Base styles — replace freely. */
:root {
  color-scheme: dark;
  --bg: #0f0d16;
  --surface: #171426;
  --text: #ece9f5;
  --muted: #a29bbd;
  --accent: #a78bfa;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}

.site-header { padding: 4rem 1.5rem 3rem; text-align: center; }
.site-header h1 { margin: 0 0 0.5rem; font-size: clamp(2rem, 5vw, 3.5rem); }
.tagline { margin: 0; color: var(--muted); }

main { max-width: 52rem; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

section {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: var(--surface);
  border: 1px solid rgba(167, 139, 250, 0.18);
  border-radius: 0.75rem;
}

section h2 { margin-top: 0; color: var(--accent); }

footer {
  padding: 2rem;
  text-align: center;
  color: var(--muted);
  border-top: 1px solid rgba(167, 139, 250, 0.12);
}
"""
    readme = f"""# {title}

{tagline}

## Run

```bash
npm install
npm run dev
```
"""
    return {
        "package.json": package,
        "index.html": index,
        "src/main.jsx": main,
        "src/App.jsx": app,
        "src/App.css": css,
        "README.md": readme,
    }


def _fastapi_files(title: str, tagline: str, sections: list[str]) -> dict[str, str]:
    esc = html.escape
    section_html = "\n".join(
        f"    <section><h2>{esc(s)}</h2><p>Add your {esc(s.lower())} content here.</p></section>"
        for s in sections
    )
    page = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(title)}</title>
    <style>
      body {{ margin: 0; font-family: system-ui, sans-serif; background: #0f0d16; color: #ece9f5; }}
      header {{ padding: 4rem 1.5rem; text-align: center; }}
      main {{ max-width: 52rem; margin: 0 auto; padding: 0 1.5rem 4rem; }}
      section {{ margin-bottom: 1.5rem; padding: 1.5rem; background: #171426; border-radius: 0.75rem; }}
      h2 {{ color: #a78bfa; margin-top: 0; }}
    </style>
  </head>
  <body>
    <header><h1>{esc(title)}</h1><p>{esc(tagline)}</p></header>
    <main>
{section_html}
    </main>
  </body>
</html>
"""
    app = '''"""Minimal FastAPI service generated by sloughGPT."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title=__TITLE__)

INDEX_HTML = """__PAGE__"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
'''
    app = app.replace("__TITLE__", repr(title)).replace("__PAGE__", page)
    requirements = "fastapi>=0.110\nuvicorn[standard]>=0.29\n"
    readme = f"""# {title}

{tagline}

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
"""
    return {"app.py": app, "requirements.txt": requirements, "README.md": readme}


def scaffold(
    template: str,
    name: str,
    title: str = "",
    tagline: str = "",
    sections: str = "",
) -> dict[str, Any]:
    """Create a project from a template and return its artifact manifest."""
    template = (template or "static").strip().lower()
    if template not in TEMPLATES:
        raise WorkspaceError(
            f"unknown template '{template}' — choose one of: {', '.join(TEMPLATES)}"
        )

    slug = slugify(name)
    title = (title or titleize(name)).strip()
    tagline = (tagline or "Built by an agent in sloughGPT.").strip()
    section_names = _sections(sections)

    if template == "vite-react":
        files = _vite_react_files(slug, title, tagline, section_names)
    elif template == "fastapi":
        files = _fastapi_files(title, tagline, section_names)
    else:
        files = _static_files(title, tagline, section_names)

    project, root = _unique_project(slug)
    for rel, content in files.items():
        dest = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dest) or root, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)

    listing = ", ".join(sorted(files))
    return {
        "artifact": True,
        "output": f'Built "{title}" ({template}) at {root}\nFiles: {listing}',
        "root": root,
        "name": project,
        "template": template,
        "title": title,
        "files": sorted(files),
        "manifest": _manifest(root),
    }


def write_file(project: str, path: str, content: str) -> dict[str, Any]:
    """Write one file inside an existing project (path-guarded)."""
    if not project:
        raise WorkspaceError("project is required — scaffold one first")
    root = _project_dir(project)
    if not os.path.isdir(root):
        raise WorkspaceError(f"project '{project}' does not exist — scaffold it first")

    rel = (path or "").strip().replace("\\", "/")
    if not rel:
        raise WorkspaceError("path is required")
    if rel.endswith("/"):
        raise WorkspaceError("path must name a file, not a directory")
    if rel.startswith("/") or re.match(r"^[a-zA-Z]:", rel):
        raise WorkspaceError("path must be relative to the project")
    dest = os.path.realpath(os.path.join(root, rel))
    if dest != root and not dest.startswith(root + os.sep):
        raise WorkspaceError("path escapes the project")
    if dest == root or os.path.isdir(dest):
        raise WorkspaceError("path must name a file, not a directory")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)

    rel = os.path.relpath(dest, root).replace(os.sep, "/")
    size = len(content.encode("utf-8"))
    return {
        "artifact": True,
        "output": f"Wrote {rel} in {project} ({size} bytes)",
        "root": root,
        "name": project,
        "path": rel,
        "bytes": size,
        "files": [rel],
        "manifest": _manifest(root),
    }


__all__ = [
    "PROJECTS_ROOT",
    "TEMPLATES",
    "WorkspaceError",
    "scaffold",
    "slugify",
    "titleize",
    "write_file",
]
