#!/bin/bash
# Generate icons for Hexagrams 8 to 64
for i in {8..64}
do
   echo "Processing Hexagram $i..."
   python3 scripts/generate_yao_icons_openai.py $i images/yao_icons
   sleep 1 # Be nice to the API
done
