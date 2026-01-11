#!/bin/bash

# Batch generation for Hexagrams 8 to 64
START=8
END=64

echo "Starting V2 batch generation from Hexagram $START to $END..."

for ((i=START; i<=END; i++))
do
    echo "Processing Hexagram $i..."
    # Using V2 script
    python3 scripts/generate_yao_icons_v2.py $i images/yao_icons
    
    # Optional: Check exit code if needed
done

echo "Batch generation completed."
