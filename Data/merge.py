import os
import re
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def collect_parts(pattern: str, root: Path) -> list[Path]:
    """
    Recursively find all files matching `pattern` under `root`.
    Sort them by the numeric part extracted from the filename.
    """
    files = list(root.rglob(pattern))
    # Extract the numeric suffix for stable sorting
    key_fn = lambda p: int(re.search(r'part(\d+)', p.name, re.I).group(1))
    return sorted(files, key=key_fn)


def merge_parts(file_list: list[Path]) -> list:
    """
    Load JSON from each file in `file_list` and return a single merged list.
    Assumes each file contains either a JSON array or a JSON object.
    Arrays are concatenated; objects are appended as list items.
    """
    merged = []
    for fp in file_list:
        with fp.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            merged.extend(data)
        else:
            merged.append(data)
    return merged


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main() -> None:
    root_dir = Path.cwd()  # run from the directory containing the parts

    # Map base names to their glob patterns
    tasks = {
        'Reasoning Enhancement': 'Reasoning Enhancement-part*.json',
        'Reflection Enhancement': 'Reflection Enhancement-part*.json',
    }

    for base_name, pattern in tasks.items():
        parts = collect_parts(pattern, root_dir)
        if not parts:
            print(f'No parts found for {base_name}, skipping.')
            continue

        print(f'Merging {len(parts)} parts for {base_name} ...')
        merged_data = merge_parts(parts)

        output_file = root_dir / f'{base_name}.json'
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print(f'✅  {output_file.name} created ({len(merged_data)} items).')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()
