#!/bin/bash

# script to delete directories without 'overall_metrics.json' in their subtree
# Usage: ./cleanup-dirs.sh /path/to/root

if [ -z "$1" ]; then
	    echo "Error: Please provide a root directory to search."
	        echo "Usage: $0 /path/to/root"
		    exit 1
fi

SEARCH_PATH="$1"

if [ ! -d "$SEARCH_PATH" ]; then
	    echo "Error: '$SEARCH_PATH' is not a valid directory."
	        exit 1
fi

# Find all directories and process them
find "$SEARCH_PATH" -type d | while read -r dir; do
    # Skip the root directory itself
        if [ "$dir" = "$SEARCH_PATH" ]; then
		        continue
			    fi
			        # Check if 'overall_metrics.json' exists in the directory or its subdirectories
				    if ! find "$dir" -name "overall_metrics.json" -type f | read -r; then
					            echo "Deleting directory (no overall_metrics.json found): $dir"
						            rm -rf "$dir"
							        fi
							done
