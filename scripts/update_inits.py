#!/usr/bin/env python3
"""Update all __init__.py files to export pl_ functions instead of pandas ones.

Strategy:
1. Read each __init__.py
2. For each `from module import func` line:
   a. Check if the module now has a pl_ version of that function
   b. Replace the import to use pl_ function name
3. Update __all__ accordingly
4. Keep special imports (CDL_PATTERN_NAMES, etc.)
"""
import re
from pathlib import Path


def get_available_pl_functions(module_path: Path) -> set[str]:
    """Get all pl_ function names defined in a module file."""
    if not module_path.exists():
        return set()
    content = module_path.read_text()
    return set(re.findall(r"^def (pl_\w+)\(", content, re.MULTILINE))


def update_init(init_path: Path, dry_run: bool = False) -> int:
    """Update a single __init__.py file. Returns count of changes."""
    content = init_path.read_text()
    lines = content.split("\n")
    new_lines = []
    changes = 0
    all_names = []

    # Special constants that aren't functions
    KEEP_AS_IS = {"CDL_PATTERN_NAMES", "ALL_PATTERNS"}

    for line in lines:
        # Match: from polars_ti.category.module import func1, func2
        match = re.match(r"from (polars_ti\.\w+\.(\w+)) import (.+)", line)
        if match:
            full_module = match.group(1)
            module_name = match.group(2)
            imports_str = match.group(3)

            # Parse individual imports
            imports = [s.strip() for s in imports_str.split(",")]
            
            # Get available pl_ functions from the module
            module_file = Path(full_module.replace(".", "/") + ".py")
            pl_funcs = get_available_pl_functions(module_file)

            new_imports = []
            for imp in imports:
                # Handle aliases: ALL_PATTERNS as CDL_PATTERN_NAMES
                alias_match = re.match(r"(\w+)\s+as\s+(\w+)", imp)
                if alias_match:
                    real_name = alias_match.group(1)
                    alias = alias_match.group(2)
                    if alias in KEEP_AS_IS or real_name in KEEP_AS_IS:
                        new_imports.append(imp)
                        all_names.append(alias)
                        continue

                # Plain import name
                func_name = imp.strip()
                
                # Check if there's a pl_ version
                pl_name = f"pl_{func_name}"
                if pl_name in pl_funcs:
                    new_imports.append(pl_name)
                    all_names.append(pl_name)
                    changes += 1
                elif func_name.startswith("pl_") and func_name in pl_funcs:
                    # Already a pl_ function
                    new_imports.append(func_name)
                    all_names.append(func_name)
                elif func_name in KEEP_AS_IS:
                    new_imports.append(func_name)
                    all_names.append(func_name)
                else:
                    # Check if the function still exists in the file
                    if module_file.exists():
                        file_content = module_file.read_text()
                        if f"def {func_name}(" in file_content:
                            new_imports.append(func_name)
                            all_names.append(func_name)
                        elif f"def pl_{func_name}(" in file_content:
                            new_imports.append(f"pl_{func_name}")
                            all_names.append(f"pl_{func_name}")
                            changes += 1
                        else:
                            # Function was removed, skip the import
                            changes += 1
                            continue
                    else:
                        new_imports.append(func_name)
                        all_names.append(func_name)

            if new_imports:
                new_lines.append(f"from {full_module} import {', '.join(new_imports)}")
            continue

        # Match __all__ block - we'll regenerate it
        if line.strip().startswith("__all__"):
            # Skip __all__ lines until we find the closing bracket
            continue
        if line.strip().startswith('"') and line.strip().endswith('",'):
            continue
        if line.strip() == "]":
            continue

        new_lines.append(line)

    # Add __all__
    if all_names:
        new_lines.append("")
        new_lines.append("__all__ = [")
        for name in sorted(set(all_names)):
            new_lines.append(f'    "{name}",')
        new_lines.append("]")
        new_lines.append("")

    new_content = "\n".join(new_lines)
    # Clean up excess blank lines
    new_content = re.sub(r"\n{3,}", "\n\n", new_content)
    if not new_content.endswith("\n"):
        new_content += "\n"

    if not dry_run:
        init_path.write_text(new_content)

    return changes


def main():
    import sys
    dry_run = "--dry-run" in sys.argv
    root = Path("polars_ti")

    categories = [
        "candles", "cycles", "momentum", "overlap", "performance",
        "statistics", "transform", "trend", "volatility", "volume",
    ]

    total_changes = 0
    for cat in categories:
        init_path = root / cat / "__init__.py"
        if not init_path.exists():
            continue
        changes = update_init(init_path, dry_run)
        total_changes += changes
        action = "Would update" if dry_run else "Updated"
        print(f"  {action}: {init_path} ({changes} changes)")

    print(f"\nTotal: {total_changes} import changes")


if __name__ == "__main__":
    main()
