"""
Data commands - Dataset management, import, and validation.
"""
import sys
import os
import json
from pathlib import Path
from typing import Optional

from domains.logging import get_global

log = get_global()
from utils.formatting import format_size, format_number


def cmd_datasets(args):
    """List available datasets."""
    datasets_dir = Path("data")
    registry_file = datasets_dir / "registry.json"
    registry = {}
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                registry = json.load(f)
        except Exception:
            pass

    log.header("Datasets")

    total_size = 0
    rows = []
    for ds in sorted(datasets_dir.iterdir()):
        if ds.is_dir():
            size = sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())
            total_size += size
            reg_info = registry.get(ds.name, {})
            vocab = reg_info.get("meta", {}).get("vocab_size", "?")
            rows.append([ds.name, format_size(size), str(vocab)])

    log.table(["Name", "Size", "Vocab"], rows)
    log.blank()
    log.key_value("Total", format_size(total_size))


def cmd_dataset_import(args, source: str):
    """Import datasets from various sources."""
    datasets_dir = Path("data")
    datasets_dir.mkdir(exist_ok=True)

    log.header(f"Import Dataset ({source})")

    if source == "github":
        url = args.url
        name = args.name or url.split("/")[-1].replace(".git", "")
        log.key_value("URL", url)
        log.key_value("Name", name)
        log.blank()

        try:
            from domains.training.data_import import RepoImporter

            repo = RepoImporter()
            result = repo.import_from_github(
                url=url,
                dataset_name=name,
                output_path=f"datasets/{name}",
            )

            if result.success:
                log.success(f"Imported {result.files_imported} files ({format_number(result.total_chars)} chars)")
                log.key_value("Location", result.output_path)
            else:
                log.error(f"Failed: {result.error}")
        except Exception as e:
            log.error(str(e))

    elif source == "hf":
        dataset_id = args.dataset_id
        name = args.name or dataset_id.split("/")[-1]
        log.key_value("Dataset ID", dataset_id)
        log.key_value("Name", name)
        log.blank()

        try:
            from domains.training.data_import import HuggingFaceImporter

            hf = HuggingFaceImporter()
            result = hf.download_dataset(
                dataset_id=dataset_id,
                name=name,
                output_dir="data",
            )

            if result.success:
                log.success(f"Imported {result.files_imported} files")
                log.key_value("Location", result.output_path)
            else:
                log.error(f"Failed: {result.error}")
        except Exception as e:
            log.error(str(e))

    elif source == "url":
        import requests

        url = args.url
        name = args.name
        output_dir = datasets_dir / name
        output_dir.mkdir(exist_ok=True)

        log.key_value("URL", url)
        log.blank()

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()

            if url.endswith(".jsonl"):
                output_file = output_dir / "corpus.jsonl"
            else:
                output_file = output_dir / "input.txt"

            with open(output_file, "w") as f:
                f.write(r.text)

            log.success(f"Downloaded {format_number(len(r.text))} chars")
            log.key_value("Location", str(output_dir))
        except Exception as e:
            log.error(str(e))


def cmd_dataset_search(args):
    """Search online for datasets."""
    query = args.query
    source = getattr(args, "source", "hf")

    log.header(f"Searching {source.upper()}")
    log.key_value("Query", query)
    log.blank()

    if source == "hf":
        try:
            from domains.training.data_import import HuggingFaceImporter
            results = HuggingFaceImporter().search_datasets(query=query, limit=args.limit)
            if results:
                log.success(f"Found {len(results)} datasets")
                log.blank()
                rows = []
                for r in results:
                    rows.append([r["id"], format_number(r.get("downloads", 0))])
                log.table(["Dataset", "Downloads"], rows)
            else:
                log.info("No results found")
        except ImportError as e:
            log.error(f"Could not search HuggingFace: {e}")
        except Exception as e:
            log.error(str(e))
    else:
        try:
            from domains.training.data_import import GitHubSearch
            results = GitHubSearch().search_repos(query=query, limit=args.limit)
            if results:
                log.success(f"Found {len(results)} repositories")
                log.blank()
                rows = []
                for r in results:
                    desc = r.get("description", "")[:60]
                    rows.append([r["full_name"], desc, str(r.get("stargazers_count", 0))])
                log.table(["Repository", "Description", "Stars"], rows)
            else:
                log.info("No results found")
        except Exception as e:
            log.error(str(e))


def cmd_data_tool(args, subcmd: str):
    """Dataset utilities - stats, validate."""
    path = Path(args.path)
    if not path.exists():
        log.error(f"Path not found: {path}")
        return

    if subcmd == "stats":
        total_lines = 0
        total_chars = 0

        if path.is_file():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        total_lines += 1
                        total_chars += len(line)

            log.header("File Statistics")
            log.key_value("Path", str(path))
            log.key_value("Lines", format_number(total_lines))
            log.key_value("Characters", format_number(total_chars))
            log.key_value("Avg Line Length", str(total_lines // max(total_lines, 1)))

        else:
            files = list(path.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            log.header("Directory Statistics")
            log.key_value("Path", str(path))
            log.key_value("Files", format_number(len([f for f in files if f.is_file()])))
            log.key_value("Total Size", format_size(total_size))

    elif subcmd == "validate":
        issues = []
        if path.is_file():
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    if not line.strip():
                        issues.append(f"Line {i}: Empty")
        else:
            files = [f for f in path.rglob("*") if f.is_file()]

        if issues:
            log.warning(f"Found {len(issues)} issues")
            for issue in issues[:10]:
                log.key_value("", issue)
        else:
            log.success("Validation passed")


def cmd_dataset_stats(args):
    """Show detailed dataset statistics."""
    name = args.name
    dataset_path = Path("data") / name

    if not dataset_path.exists():
        log.error(f"Dataset not found: {name}")
        return

    log.header(f"Dataset: {name}")

    corpus = dataset_path / "corpus.jsonl"
    input_txt = dataset_path / "input.txt"

    if corpus.exists():
        size = corpus.stat().st_size
        lines = sum(1 for _ in open(corpus))
        chars = sum(len(l) for l in open(corpus))

        log.key_value("Type", "corpus (JSONL)")
        log.key_value("Size", format_size(size))
        log.key_value("Lines", format_number(lines))
        log.key_value("Chars", format_number(chars))

        with open(corpus) as f:
            first = f.readline()
            try:
                obj = json.loads(first)
                log.key_value("Fields", ", ".join(obj.keys()))
            except (json.JSONDecodeError, ValueError):
                pass

    elif input_txt.exists():
        size = input_txt.stat().st_size
        lines = sum(1 for _ in open(input_txt))
        log.key_value("Type", "text")
        log.key_value("Size", format_size(size))
        log.key_value("Lines", format_number(lines))

    else:
        log.warning("Empty dataset")


def cmd_dataset_export(args):
    """Export dataset to zip archive."""
    import shutil

    name = args.name
    output = args.output or f"{name}.zip"

    dataset_path = Path("data") / name
    if not dataset_path.exists():
        log.error(f"Dataset not found: {name}")
        return

    log.header("Export Dataset")
    log.key_value("Name", name)
    log.key_value("Output", output)
    log.blank()

    log.step("Archiving...")
    shutil.make_archive(output.replace(".zip", ""), "zip", ".", dataset_path)
    log.success(f"Exported: {output}")


def register(subparsers):
    """Register data commands with argparse."""
    # Datasets
    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List available datasets",
    )
    datasets_parser.set_defaults(func=cmd_datasets)

    ds_sub = datasets_parser.add_subparsers(dest="ds_cmd", metavar="IMPORT")
    ds_sub.add_parser("list", help="List datasets").set_defaults(func=cmd_datasets)

    # Search
    ds_search = ds_sub.add_parser("search", help="Search online datasets")
    ds_search.add_argument("query", help="Search query")
    ds_search.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    ds_search.add_argument("--source", default="hf", choices=["hf", "github"], help="Search source")
    ds_search.set_defaults(func=cmd_dataset_search)

    # Import from GitHub
    gh_imp = ds_sub.add_parser("github", help="Import from GitHub")
    gh_imp.add_argument("url", help="GitHub repo URL")
    gh_imp.add_argument("name", nargs="?", help="Dataset name")
    gh_imp.set_defaults(func=lambda a: cmd_dataset_import(a, "github"))

    # Import from HuggingFace
    hf_imp = ds_sub.add_parser("hf", help="Import from HuggingFace")
    hf_imp.add_argument("dataset_id", help="HuggingFace dataset ID")
    hf_imp.add_argument("name", nargs="?", help="Dataset name")
    hf_imp.set_defaults(func=lambda a: cmd_dataset_import(a, "hf"))

    # URL import
    url_imp = ds_sub.add_parser("url", help="Import from URL")
    url_imp.add_argument("url", help="URL to download")
    url_imp.add_argument("name", help="Dataset name")
    url_imp.set_defaults(func=lambda a: cmd_dataset_import(a, "url"))

    # Stats
    ds_stats = ds_sub.add_parser("stats", help="Dataset statistics")
    ds_stats.add_argument("name", help="Dataset name")
    ds_stats.set_defaults(func=cmd_dataset_stats)

    # Export
    ds_export = ds_sub.add_parser("export", help="Export dataset to zip")
    ds_export.add_argument("name", help="Dataset name")
    ds_export.add_argument("--output", "-o", help="Output zip file")
    ds_export.set_defaults(func=cmd_dataset_export)

    # Data tools
    data_parser = subparsers.add_parser(
        "data",
        help="Dataset utilities",
    )
    data_sub = data_parser.add_subparsers(dest="data_cmd", metavar="SUBCOMMAND")

    data_stats_parser = data_sub.add_parser("stats", help="File/folder stats")
    data_stats_parser.add_argument("path", help="Path to inspect")
    data_stats_parser.set_defaults(func=lambda a: cmd_data_tool(a, "stats"))

    validate_parser = data_sub.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("path", help="Dataset path")
    validate_parser.set_defaults(func=lambda a: cmd_data_tool(a, "validate"))
