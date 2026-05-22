#!/usr/bin/env python3
"""Phase 5A Step 1+2: Remove all pandas imports from test files.

Strategy:
  - For files where pd. usage is ONLY inside dead parity bodies: remove the
    import and trim the dead code after pytest.skip().
  - For files where pd. also appears in fixture bodies: replace pd.Series/
    pd.DataFrame with numpy arrays / pl.DataFrame and then remove the import.
"""

import ast
import re
import sys
from pathlib import Path


def is_pandas_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("import pandas") or stripped.startswith("from pandas")


def strip_dead_parity_body(lines: list[str]) -> list[str]:
    """Remove lines after pytest.skip() inside parity test methods.

    Strips all lines between pytest.skip(...) and the next peer-level method
    or class definition (the latter handles module-level class boundaries).
    """
    result = []
    in_parity = False
    skip_seen = False
    indent_level = None

    for line in lines:
        stripped = line.strip()
        # Use the indentation of non-blank, non-comment lines only
        raw_indent = len(line) - len(line.lstrip()) if stripped else None

        # Detect entry into a parity method
        if re.match(r"\s*def test_(numerical_parity|parity_with_pandas|parity)\b", line):
            in_parity = True
            skip_seen = False
            indent_level = raw_indent
            result.append(line)
            continue

        if in_parity:
            curr_indent = raw_indent if raw_indent is not None else indent_level + 1

            # Any def or class at the same or lower indent level exits parity scope
            is_boundary = (stripped.startswith("def ") or stripped.startswith("class ")) and curr_indent <= indent_level
            if is_boundary:
                in_parity = False
                skip_seen = False
                result.append(line)
                continue

            if not skip_seen:
                result.append(line)
                if "pytest.skip(" in line:
                    skip_seen = True
                continue

            # skip_seen=True: drop all dead code until a boundary
            # (already handled boundary above, so this is pure dead-code drop)
            continue

        result.append(line)

    return result


def _replace_pd_series_call(m: re.Match) -> str:
    """Replace pd.Series(arg0, ...) with just arg0 (the array/list)."""
    inner = m.group(1)
    # Parse to isolate only the first positional argument
    # The inner content is like: "close" or "close, name='foo'" or "data=close"
    inner = inner.strip()
    if not inner:
        return "None"  # edge case: pd.Series() with no args
    # Try to split on commas not inside brackets/quotes
    # We only want the first positional arg; everything else is a named kwarg
    depth = 0
    first_arg_end = len(inner)
    for i, c in enumerate(inner):
        if c in ("(", "[", "{"):
            depth += 1
        elif c in (")", "]", "}"):
            depth -= 1
        elif c == "," and depth == 0:
            first_arg_end = i
            break
    first_arg = inner[:first_arg_end].strip()
    # If it's a keyword argument like "data=close", extract the value
    if "=" in first_arg and not first_arg.startswith("("):
        first_arg = first_arg.split("=", 1)[1].strip()
    return first_arg


def replace_pd_usage_in_fixture(content: str) -> str:
    """Replace pd.Series(x, ...) → x and pd.DataFrame({...}) → pl.DataFrame({...})."""
    # pd.DataFrame({...}) → pl.DataFrame({...})
    content = re.sub(r"\bpd\.DataFrame\b", "pl.DataFrame", content)
    # pd.Series(args) → first positional arg only
    # Use a regex that matches balanced parens in the pd.Series(...)  call.
    # We do this in a simple loop to get balanced-paren matching.
    result = []
    i = 0
    while i < len(content):
        # Try to match pd.Series( at position i
        m = re.match(r"pd\.Series\(", content[i:])
        if m:
            # Find the closing paren (balanced)
            start = i + m.end()
            depth = 1
            j = start
            while j < len(content) and depth > 0:
                if content[j] == "(":
                    depth += 1
                elif content[j] == ")":
                    depth -= 1
                j += 1
            # content[start:j-1] is the inner content of pd.Series(...)
            inner = content[start : j - 1]
            # Extract the first positional argument
            replacement = _extract_first_arg(inner)
            result.append(replacement)
            i = j
        else:
            result.append(content[i])
            i += 1
    return "".join(result)


def _extract_first_arg(inner: str) -> str:
    """Return the first positional argument from an argument list string."""
    inner = inner.strip()
    if not inner:
        return "None"
    depth = 0
    first_arg_end = len(inner)
    in_str = False
    str_char = None
    for idx, c in enumerate(inner):
        if in_str:
            if c == str_char and (idx == 0 or inner[idx - 1] != "\\"):
                in_str = False
        elif c in ("'", '"'):
            in_str = True
            str_char = c
        elif c in ("(", "[", "{"):
            depth += 1
        elif c in (")", "]", "}"):
            depth -= 1
        elif c == "," and depth == 0:
            first_arg_end = idx
            break
    first_arg = inner[:first_arg_end].strip()
    # If it looks like kwarg (e.g., data=close), extract the value
    kw_match = re.match(r"^[a-zA-Z_]\w*\s*=\s*(.+)$", first_arg)
    if kw_match:
        first_arg = kw_match.group(1).strip()
    return first_arg


def process_file(fp: Path, dry_run: bool = False) -> bool:
    """Process a single test file. Returns True if modified."""
    original = fp.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    # Check pandas presence
    has_pandas_import = any(is_pandas_import_line(l) for l in lines)
    if not has_pandas_import:
        return False

    # --- Step 1: strip dead parity bodies ---
    lines = strip_dead_parity_body(lines)
    content = "".join(lines)

    # --- Step 2: replace remaining pd. usage in fixtures ---
    content = replace_pd_usage_in_fixture(content)

    # --- Remove the pandas import line ---
    processed = []
    for line in content.splitlines(keepends=True):
        if is_pandas_import_line(line):
            continue  # drop it
        processed.append(line)
    content = "".join(processed)

    if content == original:
        return False

    if not dry_run:
        fp.write_text(content, encoding="utf-8")
        print(f"  ✓ {fp}")
    else:
        print(f"  [DRY] {fp}")
    return True


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    root = Path("tests")
    modified = 0
    for fp in sorted(root.rglob("test_*_polars.py")):
        if process_file(fp, dry_run=dry_run):
            modified += 1
    print(f"\nModified {modified} files.")


if __name__ == "__main__":
    main()
