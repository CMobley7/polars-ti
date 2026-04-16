#!/usr/bin/env python3
"""Strip legacy pandas functions from indicator files.

Each file follows this pattern:
  1. # -*- coding: utf-8 -*-
  2. Pandas imports (from pandas import ...)
  3. Utility imports for pandas (v_series, v_offset, ...)
  4. Shared @njit kernels (used by BOTH pandas and polars)
  5. def indicator_name(...):  # <- PANDAS function to REMOVE
  6. # ============================
  7. # Polars XXX Implementation
  8. # ============================
  9. import polars as pl ...
 10. def pl_indicator_name(...)

Strategy:
- Find the "# Polars ... Implementation" marker line
- Walk backward to find the separator line (# === or blank line before it)
- Keep: encoding, @njit blocks before the pandas function, Polars section
- Remove: pandas function + its imports
"""
import re
import sys
from pathlib import Path


def find_polars_marker(lines: list[str]) -> int:
    """Find the line containing 'Polars ... Implementation'."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "Polars" in stripped and "Implementation" in stripped and stripped.startswith("#"):
            return i
    return -1


def find_section_start(lines: list[str], marker_idx: int) -> int:
    """Walk backward from the marker to find the section separator start."""
    i = marker_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("# ===") or stripped.startswith("# ---"):
            return i
        if stripped == "" or stripped.startswith("#"):
            i -= 1
            continue
        # Hit actual code - section starts at the next line
        return i + 1
    return marker_idx


def find_njit_blocks(lines: list[str], before_idx: int) -> list[tuple[int, int]]:
    """Find all @njit-decorated function blocks before the given index."""
    blocks = []
    i = 0
    while i < before_idx:
        stripped = lines[i].strip()
        if stripped.startswith("@njit") or stripped.startswith("@_njit"):
            block_start = i
            # Move to function def
            i += 1
            while i < before_idx and not lines[i].strip().startswith("def "):
                i += 1
            if i >= before_idx:
                break
            # Move past function body (indented lines)
            i += 1
            while i < before_idx:
                if lines[i].strip() == "":
                    # Check if next non-blank is still indented
                    j = i + 1
                    while j < before_idx and lines[j].strip() == "":
                        j += 1
                    if j >= before_idx or (not lines[j].startswith(" ") and not lines[j].startswith("\t")):
                        i = j
                        break
                    i = j
                elif not lines[i].startswith(" ") and not lines[i].startswith("\t"):
                    break
                else:
                    i += 1
            blocks.append((block_start, i))
        else:
            i += 1
    return blocks


def get_kernel_numpy_imports(kernel_text: str) -> set[str]:
    """Determine which numpy names a kernel uses."""
    names = set()
    numpy_names = [
        "nan", "zeros", "zeros_like", "isnan", "floor", "roll",
        "arctan", "rad2deg", "empty", "float64", "copy", "exp",
        "cos", "sqrt", "log", "log10", "full", "ones", "abs",
        "maximum", "minimum", "where", "array", "arange",
    ]
    for name in numpy_names:
        if re.search(rf"\b{name}\b", kernel_text):
            names.add(name)
    return names


def strip_file(filepath: Path, dry_run: bool = False) -> dict:
    """Strip pandas code from a single file. Returns info dict."""
    content = filepath.read_text()
    lines = content.split("\n")

    marker_idx = find_polars_marker(lines)
    if marker_idx < 0:
        return {"status": "skip", "reason": "no Polars marker"}

    section_start = find_section_start(lines, marker_idx)
    njit_blocks = find_njit_blocks(lines, section_start)

    # Find the pandas function definition (first `def ` that's NOT `def pl_` and NOT @njit)
    pandas_fn_name = None
    for i in range(section_start):
        if lines[i].startswith("def ") and not lines[i].startswith("def pl_"):
            pandas_fn_name = lines[i].split("(")[0].replace("def ", "").strip()
            break

    # Build new file content
    new_lines = []

    # 1. Keep encoding comment
    if lines[0].startswith("# -*-"):
        new_lines.append(lines[0])

    # 2. Collect imports needed by @njit kernels
    if njit_blocks:
        # Check what the kernels need
        all_kernel_text = ""
        for start, end in njit_blocks:
            all_kernel_text += "\n".join(lines[start:end]) + "\n"

        np_names = get_kernel_numpy_imports(all_kernel_text)
        if np_names:
            # Use the broader import pattern from the original file
            # Find original numpy import line to preserve style
            has_np_import = False
            for line in lines[:section_start]:
                if line.startswith("from numpy import "):
                    new_lines.append(line)
                    has_np_import = True
                    break
            if not has_np_import and np_names:
                new_lines.append("from numpy import " + ", ".join(sorted(np_names)))

        new_lines.append("from numba import njit")
        new_lines.append("")
        new_lines.append("")

        # 3. Add @njit kernel blocks
        for start, end in njit_blocks:
            new_lines.extend(lines[start:end])
            if not new_lines[-1].strip() == "":
                new_lines.append("")
            new_lines.append("")

    # 4. Add the Polars section (from separator to end)
    polars_section = lines[section_start:]
    # Strip leading blank lines from polars section
    while polars_section and polars_section[0].strip() == "":
        polars_section.pop(0)
    new_lines.extend(polars_section)

    # Ensure file ends with single newline
    new_content = "\n".join(new_lines)
    new_content = re.sub(r"\n{4,}", "\n\n\n", new_content)
    if not new_content.endswith("\n"):
        new_content += "\n"

    info = {
        "status": "ok",
        "polars_start": section_start + 1,
        "kernels": len(njit_blocks),
        "pandas_fn": pandas_fn_name,
        "orig_lines": len(lines),
        "new_lines": len(new_content.split("\n")),
    }

    if dry_run:
        return info

    filepath.write_text(new_content)
    return info


def main():
    dry_run = "--dry-run" in sys.argv
    root = Path("polars_ti")

    categories = [
        "candles", "cycles", "momentum", "overlap", "performance",
        "statistics", "transform", "trend", "volatility", "volume",
    ]

    results = {"ok": 0, "skip": 0, "errors": []}

    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        for filepath in sorted(cat_dir.glob("*.py")):
            if filepath.name == "__init__.py":
                continue
            try:
                info = strip_file(filepath, dry_run)
                if info["status"] == "ok":
                    results["ok"] += 1
                    if dry_run:
                        print(f"  {filepath}: L{info['polars_start']}, "
                              f"{info['kernels']} kernels, "
                              f"pd_fn={info['pandas_fn']}, "
                              f"{info['orig_lines']}->{info['new_lines']} lines")
                else:
                    results["skip"] += 1
                    print(f"  SKIP: {filepath} ({info.get('reason', '?')})")
            except Exception as e:
                results["errors"].append((str(filepath), str(e)))
                print(f"  ERROR: {filepath}: {e}")

    print(f"\nDone: {results['ok']} stripped, {results['skip']} skipped, "
          f"{len(results['errors'])} errors")
    if results["errors"]:
        for fp, err in results["errors"]:
            print(f"  {fp}: {err}")


if __name__ == "__main__":
    main()
