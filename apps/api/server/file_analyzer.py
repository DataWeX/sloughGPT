"""Proactive file analyzer — detect type, extract patterns, surface insights.

When a file is uploaded, this module analyzes its content and generates
proactive insights BEFORE the user asks any questions. The AI should
notice things, not just wait.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FileInsight:
    """A single insight extracted from a file."""

    category: str  # "structure", "content", "warning", "summary"
    label: str  # short human-readable label
    detail: str  # the actual insight text
    priority: int = 0  # higher = more important


@dataclass
class FileAnalysis:
    """Complete analysis of a file."""

    file_type: str  # "code", "data", "document", "config", "text"
    language: str = ""  # detected language if code
    line_count: int = 0
    word_count: int = 0
    char_count: int = 0
    insights: list[FileInsight] = field(default_factory=list)
    summary: str = ""
    suggested_questions: list[str] = field(default_factory=list)


# ── Pattern detectors ─────────────────────────────────────────────────────────

_CODE_EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "React", ".tsx": "React TS", ".rs": "Rust",
    ".go": "Go", ".java": "Java", ".c": "C", ".cpp": "C++",
    ".h": "C Header", ".rb": "Ruby", ".php": "PHP",
    ".sh": "Shell", ".bash": "Shell", ".sql": "SQL",
}

_DATA_EXTENSIONS = {
    ".json": "JSON", ".jsonl": "JSON Lines", ".csv": "CSV",
    ".yaml": "YAML", ".yml": "YAML", ".xml": "XML",
    ".toml": "TOML", ".ini": "INI",
}

_DOC_EXTENSIONS = {
    ".md": "Markdown", ".txt": "Plain Text", ".html": "HTML",
    ".htm": "HTML", ".rst": "reStructuredText",
}


def _detect_type(ext: str) -> str:
    ext = ext.lower()
    if ext in _CODE_EXTENSIONS:
        return "code"
    if ext in _DATA_EXTENSIONS:
        return "data"
    if ext in _DOC_EXTENSIONS:
        return "document"
    if ext in (".pdf", ".docx", ".doc"):
        return "document"
    return "text"


def _count_words(text: str) -> int:
    return len(text.split())


def _extract_code_insights(text: str, language: str) -> list[FileInsight]:
    """Extract insights from code files."""
    insights = []
    lines = text.split("\n")

    # Functions and classes
    funcs = [l.strip() for l in lines if re.match(r"\s*(def |function |fn |func |async def )", l)]
    classes = [l.strip() for l in lines if re.match(r"\s*(class |struct |interface |enum )", l)]
    imports = [l.strip() for l in lines if re.match(r"\s*(import |from |use |require|#include)", l)]

    if funcs:
        insights.append(FileInsight(
            category="structure",
            label=f"{len(funcs)} function{'s' if len(funcs) != 1 else ''}",
            detail=f"Defines: {', '.join(f.split('(')[0].split()[-1] for f in funcs[:5])}" +
                   (f" and {len(funcs) - 5} more" if len(funcs) > 5 else ""),
            priority=2,
        ))
    if classes:
        def _class_name(c: str) -> str:
            parts = c.replace(":", " ").replace("(", " ").split()
            for p in parts:
                if p not in ("class", "struct", "interface", "enum", "extends", "implements"):
                    return p
            return c[:30]
        insights.append(FileInsight(
            category="structure",
            label=f"{len(classes)} class{'es' if len(classes) != 1 else ''}",
            detail=f"Defines: {', '.join(_class_name(c) for c in classes[:5])}",
            priority=2,
        ))
    if imports:
        external = [i for i in imports if not i.startswith("from .") and not i.startswith("from domain")]
        if external:
            insights.append(FileInsight(
                category="content",
                label=f"{len(external)} import{'s' if len(external) != 1 else ''}",
                detail=f"Dependencies: {', '.join(i.split()[-1].split('.')[0] for i in external[:5])}",
                priority=1,
            ))

    # TODOs and FIXMEs
    todos = [l.strip() for l in lines if "TODO" in l or "FIXME" in l or "HACK" in l]
    if todos:
        insights.append(FileInsight(
            category="warning",
            label=f"{len(todos)} TODO/FIXME",
            detail=todos[0][:120] + ("..." if len(todos[0]) > 120 else ""),
            priority=3,
        ))

    # Long functions (potential complexity)
    func_starts = [i for i, l in enumerate(lines) if re.match(r"\s*(def |function |fn |func )", l)]
    for start in func_starts[:3]:
        end = next((i for i in range(start + 1, min(start + 100, len(lines)))
                     if lines[i] and not lines[i][0].isspace()), len(lines))
        if end - start > 50:
            name = lines[start].strip().split("(")[0].split()[-1]
            insights.append(FileInsight(
                category="warning",
                label=f"Long function: {name}()",
                detail=f"{end - start} lines — consider breaking it up",
                priority=2,
            ))

    return insights


def _extract_data_insights(text: str, language: str) -> list[FileInsight]:
    """Extract insights from data files."""
    insights = []
    lines = text.strip().split("\n")

    if language == "JSON":
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                insights.append(FileInsight(
                    category="structure",
                    label=f"{len(data)} records",
                    detail=f"Array of {len(data)} items" +
                           (f", keys: {', '.join(list(data[0].keys())[:5])}" if data and isinstance(data[0], dict) else ""),
                    priority=2,
                ))
            elif isinstance(data, dict):
                insights.append(FileInsight(
                    category="structure",
                    label=f"{len(data)} keys",
                    detail=f"Object with keys: {', '.join(list(data.keys())[:8])}",
                    priority=2,
                ))
        except Exception:
            insights.append(FileInsight(category="structure", label="Invalid JSON", detail="Could not parse", priority=3))

    elif language == "CSV":
        header = lines[0] if lines else ""
        cols = header.split(",")
        insights.append(FileInsight(
            category="structure",
            label=f"{len(cols)} columns, {len(lines) - 1} rows",
            detail=f"Columns: {', '.join(c.strip() for c in cols[:6])}" +
                   (f" and {len(cols) - 6} more" if len(cols) > 6 else ""),
            priority=2,
        ))

    elif language == "JSON Lines":
        count = len(lines)
        insights.append(FileInsight(
            category="structure",
            label=f"{count} records",
            detail=f"JSON Lines file with {count} entries",
            priority=2,
        ))

    return insights


def _extract_document_insights(text: str, language: str) -> list[FileInsight]:
    """Extract insights from document files."""
    insights = []
    lines = text.strip().split("\n")

    # Headings
    headings = [l.strip() for l in lines if re.match(r"^#{1,6}\s", l)]
    if headings:
        insights.append(FileInsight(
            category="structure",
            label=f"{len(headings)} section{'s' if len(headings) != 1 else ''}",
            detail=f"Structure: {' → '.join(h.lstrip('#').strip() for h in headings[:5])}",
            priority=2,
        ))

    # Links
    links = re.findall(r"https?://[^\s\)]+", text)
    if links:
        insights.append(FileInsight(
            category="content",
            label=f"{len(links)} link{'s' if len(links) != 1 else ''}",
            detail=f"References: {', '.join(links[:3])}" +
                   (f" and {len(links) - 3} more" if len(links) > 3 else ""),
            priority=1,
        ))

    # Code blocks in markdown
    code_blocks = re.findall(r"```(\w*)", text)
    if code_blocks:
        insights.append(FileInsight(
            category="content",
            label=f"{len(code_blocks)} code block{'s' if len(code_blocks) != 1 else ''}",
            detail=f"Languages: {', '.join(set(code_blocks) - {''})}" if any(code_blocks) else "Unlabeled blocks",
            priority=1,
        ))

    # Key terms (bold/italic in markdown)
    bold_terms = re.findall(r"\*\*([^*]+)\*\*", text)
    if bold_terms:
        insights.append(FileInsight(
            category="content",
            label=f"{len(bold_terms)} emphasized term{'s' if len(bold_terms) != 1 else ''}",
            detail=f"Key terms: {', '.join(bold_terms[:5])}",
            priority=1,
        ))

    # Dates
    dates = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    if dates:
        insights.append(FileInsight(
            category="content",
            label=f"{len(dates)} date{'s' if len(dates) != 1 else ''}",
            detail=f"Found dates: {', '.join(dates[:5])}",
            priority=1,
        ))

    # Email addresses
    emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
    if emails:
        insights.append(FileInsight(
            category="content",
            label=f"{len(emails)} email{'s' if len(emails) != 1 else ''}",
            detail=f"Contacts: {', '.join(emails[:3])}",
            priority=1,
        ))

    return insights


def _generate_suggested_questions(analysis: FileAnalysis, text: str) -> list[str]:
    """Generate smart suggested questions based on file content."""
    questions = []
    ft = analysis.file_type

    if ft == "code":
        questions.append("Explain what this code does")
        questions.append("Are there any bugs or issues?")
        if "TODO" in text or "FIXME" in text:
            questions.append("What needs to be fixed?")
        questions.append("How would you improve this?")

    elif ft == "data":
        questions.append("Summarize this data")
        questions.append("What patterns do you see?")
        questions.append("What are the outliers?")

    elif ft == "document":
        questions.append("Summarize this document")
        questions.append("What are the key points?")
        questions.append("Explain this in simple terms")

    else:
        questions.append("Summarize this file")
        questions.append("What are the key points?")
        questions.append("Explain this in simple terms")

    # Ensure we always have 3 questions
    while len(questions) < 3:
        generic = ["What's important here?", "Give me an overview", "What should I know?"]
        for q in generic:
            if q not in questions:
                questions.append(q)
                break
        break

    return questions[:3]


def analyze_file(filename: str, content: str) -> FileAnalysis:
    """Analyze a file and extract proactive insights.

    This is the main entry point. Call it when a file is uploaded
    to get insights BEFORE the user asks anything.
    """
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    file_type = _detect_type(ext)
    language = ""
    if file_type == "code":
        language = _CODE_EXTENSIONS.get(ext, "")
    elif file_type == "data":
        language = _DATA_EXTENSIONS.get(ext, "")
    elif file_type == "document":
        language = _DOC_EXTENSIONS.get(ext, "")

    lines = content.split("\n")
    analysis = FileAnalysis(
        file_type=file_type,
        language=language,
        line_count=len(lines),
        word_count=_count_words(content),
        char_count=len(content),
    )

    # Extract type-specific insights
    if file_type == "code":
        analysis.insights = _extract_code_insights(content, language)
    elif file_type == "data":
        analysis.insights = _extract_data_insights(content, language)
    elif file_type == "document":
        analysis.insights = _extract_document_insights(content, language)

    # Sort by priority
    analysis.insights.sort(key=lambda x: -x.priority)

    # Generate summary
    parts = []
    if analysis.line_count:
        parts.append(f"{analysis.line_count} lines")
    if analysis.word_count:
        parts.append(f"{analysis.word_count} words")
    if language:
        parts.append(language)
    analysis.summary = ", ".join(parts) if parts else f"{analysis.char_count} characters"

    # Generate smart suggested questions
    analysis.suggested_questions = _generate_suggested_questions(analysis, content)

    return analysis
