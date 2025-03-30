#!/bin/bash

PRESSURE=1
MODE=0nubb

input_file=/home/argon/Projects/Krishan/ATPC/eventlists/ATPC_${MODE}_${PRESSURE}bar_nexusfiles.txt
echo "Input File: $input_file"

NUM_LINES=$(wc -l < $input_file)

echo "Number of lines to process $NUM_LINES"


sbatch --array=1-1000            RunGetTrueInfo.sh $PRESSURE $MODE
sbatch --array=1001-2000         RunGetTrueInfo.sh $PRESSURE $MODE
sbatch --array=2001-3000         RunGetTrueInfo.sh $PRESSURE $MODE
sbatch --array=3000-"$NUM_LINES" RunGetTrueInfo.sh $PRESSURE $MODE


