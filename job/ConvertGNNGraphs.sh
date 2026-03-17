#!/bin/bash
#SBATCH -J GNN # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 500 # Memory request (6Gb)
#SBATCH -t 0-6:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/GNN_%A_%a.out # Standard output
#SBATCH -e log/GNN_%A_%a.err # Standard error


# to run:
# sbatch --array=1-98 ConvertGNNGraphs.sh

start=`date +%s`

# Setup nexus and run
echo "Setting up environment"
source /home/argon/Projects/Krishan/venv/bin/activate


# MODE=0nubb
# MODE=single
# MODE=Bi
MODE=Tl

BATCH_SIZE=100

echo "Mode is: ${MODE}"

echo "python3 /home/argon/Projects/Krishan/ATPC/scripts/Convert_to_Graph.py ${SLURM_ARRAY_TASK_ID} ${BATCH_SIZE} ${MODE}"
python3 /home/argon/Projects/Krishan/ATPC/scripts/Convert_to_Graph.py ${SLURM_ARRAY_TASK_ID} ${BATCH_SIZE} ${MODE}

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 