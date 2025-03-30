#!/bin/bash


P=1
MODE="0nubb

input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/ATPC/eventlists/ATPC_${MODE}_${PRESSURE}bar_nexusfiles.txt)
echo "Input File: $input_file"

NUM_LINES=$(wc -l < $input_file)

echo $NUM_LINES

#sbatch --array=1-"$NUM_LINES" RunGetTrueInfo.sh $P $MODE



