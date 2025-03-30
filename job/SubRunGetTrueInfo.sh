#!/bin/bash

PRESSURE=1
MODE=0nubb

input_file=/home/argon/Projects/Krishan/ATPC/eventlists/ATPC_${MODE}_${PRESSURE}bar_nexusfiles.txt
echo "Input File: $input_file"

NUM_LINES=$(wc -l < $input_file)
EXTRALINES=$((NUM_LINES - 3000))

echo "Number of extra lines to process $EXTRALINES"

sbatch --array=1-1000 RunGetTrueInfo.sh $PRESSURE $MODE 0
sbatch --array=1-1000 RunGetTrueInfo.sh $PRESSURE $MODE 1000
sbatch --array=1-1000 RunGetTrueInfo.sh $PRESSURE $MODE 2000

# Only submit if we have more than 3k files
if (( EXTRALINES > 0 )); then
    sbatch --array=1-"$EXTRALINES" RunGetTrueInfo.sh $PRESSURE $MODE 3000
fi


