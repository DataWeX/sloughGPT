"""CLI interface for AML — parse, validate, and convert AML files."""

from __future__ import annotations

import argparse
import json
import sys

from aml.parser import parse, parse_file
from aml.serializer import serialize, dict_to_aml


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse AML and print the AST summary."""
    doc = parse_file(args.file)
    print(f"AML {doc.version}  ({len(doc.blocks)} blocks, {len(doc.errors)} errors)")
    for b in doc.blocks:
        name = f"  {b.name}" if b.name else ""
        meta = f"  ({len(b.metadata)} keys)" if b.metadata else ""
        print(f"  @{b.tag}{name}{meta}  [line {b.line}]")
    if doc.errors:
        print("\nErrors:")
        for e in doc.errors:
            print(f"  {e}")
    return 0


def cmd_to_json(args: argparse.Namespace) -> int:
    """Convert AML to JSON."""
    doc = parse_file(args.file)
    out = doc.to_dict()
    if args.pretty:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_to_aml(args: argparse.Namespace) -> int:
    """Convert JSON to AML."""
    data = json.load(sys.stdin)
    print(dict_to_aml(data))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate AML syntax."""
    doc = parse_file(args.file)
    if doc.errors:
        for e in doc.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK — AML {doc.version}, {len(doc.blocks)} blocks")
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    """Re-format AML (normalize whitespace)."""
    doc = parse_file(args.file)
    print(serialize(doc), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aml",
        description="AML — Automatic Markup Language toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    p_parse = sub.add_parser("parse", help="Parse and summarize an AML file")
    p_parse.add_argument("file", help="AML file to parse")

    p_json = sub.add_parser("to-json", help="Convert AML to JSON")
    p_json.add_argument("file", help="AML file to convert")
    p_json.add_argument("--pretty", "-p", action="store_true", help="Pretty-print JSON")

    p_aml = sub.add_parser("from-json", help="Convert JSON (stdin) to AML")

    p_val = sub.add_parser("validate", help="Validate AML syntax")
    p_val.add_argument("file", help="AML file to validate")

    p_fmt = sub.add_parser("format", help="Re-format AML")
    p_fmt.add_argument("file", help="AML file to format")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "parse": cmd_parse,
        "to-json": cmd_to_json,
        "from-json": cmd_to_aml,
        "validate": cmd_validate,
        "format": cmd_format,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
