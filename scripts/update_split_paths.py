#!/usr/bin/env python3
"""Utility for rewriting dataset paths inside NCDB split JSON files.

This script walks over all JSON files inside a provided directory and updates
any path fields (e.g. ``dataset_root`` or ``image_path``) that contain a given
substring. It's primarily intended to fix cases where the split metadata still
points to a low-resolution dataset copy (``ncdb-cls-640x384``) after the
high-resolution images were copied into a new location
(``ncdb-cls-1920x1536-img-only``).

Example
-------
::

    python scripts/update_split_paths.py \
        --splits-dir "D:/data/ncdb-cls/ncdb-cls-1920x1536-img-only/splits" \
        --old "ncdb-cls-640x384" \
        --new "ncdb-cls-1920x1536-img-only"

The command above rewrites every occurrence of ``ncdb-cls-640x384`` to
``ncdb-cls-1920x1536-img-only`` in the default set of keys. Run with
``--dry-run`` first to inspect the planned changes without touching the files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

# Default keys we rewrite. Users can extend this list via CLI.
DEFAULT_KEYS: Tuple[str, ...] = (
    "dataset_root",
    "image_path",
    "a5_original_path",
    "a6_original_path",
    "pcd_original_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite dataset paths inside NCDB split JSON files.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing split JSON files to update.")
    parser.add_argument(
        "--old",
        "--old-substring",
        dest="old",
        action="append",
        required=True,
        help="Substring to replace. May be provided multiple times.")
    parser.add_argument(
        "--new",
        "--new-substring",
        dest="new",
        action="append",
        required=True,
        help="Replacement substring. Must appear the same number of times as --old.")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=list(DEFAULT_KEYS),
        help=("JSON keys to update. Defaults to %(default)s. "
              "Multiple values can be provided."))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying files.")
    parser.add_argument(
        "--backup",
        action="store_true",
        default=False,
        help="Create a .bak copy before writing each file.")
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for reading/writing JSON files (default: %(default)s).")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level when rewriting files (default: %(default)d).")
    return parser.parse_args()


def _build_replacements(old_list: Iterable[str], new_list: Iterable[str]) -> List[Tuple[str, str]]:
    old_list = list(old_list)
    new_list = list(new_list)
    if len(old_list) != len(new_list):
        raise ValueError("--old and --new must be provided the same number of times")
    return list(zip(old_list, new_list))


def _replace_substrings(value: str, replacements: Iterable[Tuple[str, str]]) -> str:
    result = value
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new)
    return result


def _update_entry(entry: dict, keys: Iterable[str], replacements: Iterable[Tuple[str, str]]) -> bool:
    changed = False
    for key in keys:
        if key not in entry:
            continue
        value = entry[key]
        if not isinstance(value, str):
            continue
        new_value = _replace_substrings(value, replacements)
        if new_value != value:
            entry[key] = new_value
            changed = True
    return changed


def process_file(
    json_path: Path,
    keys: Iterable[str],
    replacements: Iterable[Tuple[str, str]],
    *,
    dry_run: bool,
    backup: bool,
    encoding: str,
    indent: int,
) -> Tuple[int, int]:
    with json_path.open("r", encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} does not contain a list of entries")

    modified_entries = 0
    modified_fields = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        before = dict(entry)
        if _update_entry(entry, keys, replacements):
            modified_entries += 1
            for key in keys:
                if key in before and before[key] != entry.get(key):
                    modified_fields += 1

    if modified_entries and not dry_run:
        if backup:
            shutil.copy2(json_path, json_path.with_suffix(json_path.suffix + ".bak"))
        with json_path.open("w", encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.write("\n")

    return modified_entries, modified_fields


def main() -> None:
    args = parse_args()
    replacements = _build_replacements(args.old, args.new)
    keys = tuple(args.keys)

    json_files = sorted(Path(args.splits_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.splits_dir}")

    total_entries = 0
    total_fields = 0
    for json_path in json_files:
        entries, fields = process_file(
            json_path,
            keys,
            replacements,
            dry_run=args.dry_run,
            backup=args.backup,
            encoding=args.encoding,
            indent=args.indent,
        )
        if entries:
            action = "would update" if args.dry_run else "updated"
            print(f"{action} {entries} entries ({fields} fields) in {json_path}")
        else:
            print(f"no changes needed for {json_path}")
        total_entries += entries
        total_fields += fields

    summary_action = "would update" if args.dry_run else "updated"
    print(
        f"{summary_action} {total_entries} entries across {len(json_files)} files "
        f"({total_fields} fields)."
    )


if __name__ == "__main__":
    main()
