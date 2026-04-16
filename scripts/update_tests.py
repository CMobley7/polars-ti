#!/usr/bin/env python3
"""Update test files that import old pandas indicator functions.

Strategy:
1. Find 'from polars_ti.X.Y import pandas_func, pl_func' lines
2. Remove the pandas_func from the import
3. Comment out or skip parity tests that use the pandas function
4. Preserve all pl_-only tests unchanged
"""
import re
import sys
from pathlib import Path


def update_test_file(filepath: Path, dry_run: bool = False) -> int:
    """Update a single test file. Returns number of changes made."""
    content = filepath.read_text()
    changes = 0

    # Match: from polars_ti.X.Y import func1, pl_func
    # or:    from polars_ti.X.Y import func1, pl_func, func2
    def fix_import(m):
        nonlocal changes
        module = m.group(1)
        imports_str = m.group(2)
        imports = [i.strip() for i in imports_str.split(",")]
        # Keep only pl_ imports and module-level constants
        new_imports = [i for i in imports if i.startswith("pl_") or i.isupper()]
        if len(new_imports) == len(imports):
            return m.group(0)  # no change
        if not new_imports:
            changes += 1
            return f"# {m.group(0)}  # REMOVED: pandas func removed"
        changes += 1
        return f"from {module} import {', '.join(new_imports)}"

    content = re.sub(
        r"from (polars_ti\.\w+\.\w+) import ([^\n]+)",
        fix_import,
        content,
    )

    # Comment out pandas imports
    def fix_pandas_import(m):
        nonlocal changes
        changes += 1
        return f"# {m.group(0)}  # REMOVED: pandas dependency"

    content = re.sub(
        r"^import pandas.*$",
        fix_pandas_import,
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^from pandas.*$",
        fix_pandas_import,
        content,
        flags=re.MULTILINE,
    )

    # Fix tests that call old pandas function (e.g. mom(pd_close, ...)) 
    # by skipping those test methods or converting them to pl_ calls
    # Strategy: add pytest.importorskip-style skip for tests using old funcs
    # Simple approach: find test methods that reference non-pl_ indicator calls
    # and add a skip marker explaining the pandas impl was removed

    if not dry_run and changes > 0:
        filepath.write_text(content)

    return changes


def main():
    dry_run = "--dry-run" in sys.argv

    test_dirs = [
        "tests/momentum",
        "tests/overlap",
        "tests/trend",
        "tests/volatility",
        "tests/volume",
        "tests/candles",
        "tests/cycles",
        "tests/statistics",
        "tests/transform",
        "tests/performance",
    ]

    total = 0
    for td in test_dirs:
        p = Path(td)
        if not p.exists():
            continue
        for filepath in sorted(p.glob("*.py")):
            if filepath.name == "__init__.py":
                continue
            changes = update_test_file(filepath, dry_run)
            if changes:
                action = "Would update" if dry_run else "Updated"
                print(f"  {action}: {filepath} ({changes} changes)")
            total += changes

    print(f"\nTotal: {total} changes")


if __name__ == "__main__":
    main()
