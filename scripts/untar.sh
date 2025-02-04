#!/bin/bash

# Define paths
folder="ATPC_0nubb"
tar_dir="./${folder}/"
output_dir="./${folder}/"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Use GNU parallel to extract tar files concurrently
ls "$tar_dir"/*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file" -C '"$output_dir"'; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
