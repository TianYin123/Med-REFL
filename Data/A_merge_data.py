import json
import os
import re
from glob import iglob
from collections import defaultdict

def collect_parts(root_dir: str):
    """
    Returns {prefix: sorted list of paths} for all '*-part*.json' files.
    """
    pattern = re.compile(r'^(?P<prefix>.+)-part(?P<num>\d+)\.json$', re.IGNORECASE)
    groups = defaultdict(list)

    for path in iglob(os.path.join(root_dir, '*.json')):
        match = pattern.match(os.path.basename(path))
        if match:
            groups[match.group('prefix')].append(path)

    # Sort each group by numeric part number
    for prefix, paths in groups.items():
        paths.sort(key=lambda p: int(pattern.match(os.path.basename(p)).group('num')))
    return groups


def merge_array(parts):
    """
    Concatenates all arrays contained in the given JSON files.
    """
    merged = []
    for file in parts:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{file} is not a top-level array.")
        merged.extend(data)
    return merged


def main():
    # Directory that holds this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    groups = collect_parts(script_dir)

    if not groups:
        print("No part files found.")
        return

    for prefix, parts in groups.items():
        print(f"Merging {prefix} from {len(parts)} parts...")
        try:
            merged = merge_array(parts)
            out_file = os.path.join(script_dir, f"{prefix}.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved {out_file}")
        except Exception as e:
            print(f"  !! Error merging {prefix}: {e}")


if __name__ == "__main__":
    main()
