# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import os
import argparse
import re

# Thank you copilot
PATTERN = re.compile(r'_(\d+)\.(jpg|tif)$', re.IGNORECASE)

def find_matches(root_dir):
    matches = []
    for f in os.listdir(root_dir):
        path = os.path.join(root_dir, f)
        if os.path.isfile(path) and PATTERN.search(f):
            matches.append(path)
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--dry-run", action="store_true", help="Don't delete anything, just list what would be deleted.",)
    args = parser.parse_args()
    
    matches = find_matches(args.data_path)
    if not matches:
        print("No patch files found.")

    for p in matches:
        print("\t" + p)

    if args.dry_run:
        print("\nDry run complete. No files were deleted.", len(matches))
        return

    deleted = 0
    for p in matches:
        try:
            os.remove(p)
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {p}: {e}", file=sys.stderr)

    print(f"\nDeleted {deleted} files.")

if __name__ == "__main__":
    main()

