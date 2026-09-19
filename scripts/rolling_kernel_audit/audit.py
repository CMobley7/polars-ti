"""Conservative, reproducible AST inventory; flags require complexity review."""

import argparse
import ast
import json
import subprocess
from pathlib import Path

AGGREGATES = {"sum", "mean", "max", "min", "std", "var", "median", "sort", "argsort", "dot", "all", "any"}
HIDDEN = {"convolve", "correlate", "rolling_map", "rolling_apply"}


def inspect_source(source: str) -> list[dict[str, str | int | list[str]]]:
    """Inventory every function, including negatives and locally aliased slices."""
    tree = ast.parse(source)
    results = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        reasons: list[str] = []
        aliases = {
            target.id
            for node in ast.walk(function)
            if isinstance(node, ast.Assign) and any(isinstance(child, ast.Slice) for child in ast.walk(node.value))
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(function):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child is node:
                        continue
                    if isinstance(child, (ast.For, ast.While)):
                        bound = ast.unparse(child.iter if isinstance(child, ast.For) else child.test)
                        reasons.append(f"nested loop L{child.lineno}: {bound}")
                    if isinstance(child, ast.Call):
                        name = ast.unparse(child.func).split(".")[-1]
                        sliced = any(isinstance(part, ast.Slice) for part in ast.walk(child))
                        aliased = any(isinstance(part, ast.Name) and part.id in aliases for part in ast.walk(child))
                        if sliced or aliased:
                            reasons.append(f"window aggregation L{child.lineno}: {ast.unparse(child)}")
            if isinstance(node, ast.For) and function.name in {"increasing", "decreasing", "pascals_triangle"}:
                reasons.append(f"parameter-sized construction L{node.lineno}: {ast.unparse(node.iter)}")
            if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in HIDDEN:
                reasons.append(f"hidden rolling work L{node.lineno}: {ast.unparse(node.func)}")
        results.append(
            {
                "function": function.name,
                "line": function.lineno,
                "status": "flagged" if reasons else "clear",
                "reasons": sorted(set(reasons)),
            }
        )
    return results


def main() -> None:
    """Sweep every tracked Python file plus new source/tests, including negatives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = Path(__file__).with_name("baseline_commit.sha").read_text().strip()
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True).splitlines()
    paths = {path for path in tracked if path.endswith(".py")}
    if not args.baseline:
        for directory in ["polars_ti", "tests", "scripts"]:
            paths.update(
                str(path.relative_to(root))
                for path in (root / directory).rglob("*.py")
                if not any(part.startswith(".") for part in path.relative_to(root).parts)
            )
    records = []
    for relative in sorted(paths):
        source = (
            subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root, text=True)
            if args.baseline
            else (root / relative).read_text()
        )
        records.extend({"file": relative, **record} for record in inspect_source(source))
    print(
        json.dumps(
            {
                "files_scanned": len(paths),
                "production_files": sum(path.startswith("polars_ti/") for path in paths),
                "functions": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
