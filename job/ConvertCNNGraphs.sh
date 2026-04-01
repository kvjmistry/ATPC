#!/bin/bash
#SBATCH -J GNN # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 500 # Memory request (6Gb)
#SBATCH -t 0-6:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/CNN_%A_%a.out # Standard output
#SBATCH -e log/CNN_%A_%a.err # Standard error

# to run:
# sbatch --array=1-100%5 ConvertCNNGraphs.sh

start=`date +%s`

# Setup nexus and run
echo "Setting up environment"
source /home/argon/Projects/Krishan/venv/bin/activate


BATCH_SIZE=100

echo "python3 /home/argon/Projects/Krishan/ATPC/scripts/Convert_to_CNNpt.py ${SLURM_ARRAY_TASK_ID} ${BATCH_SIZE}"
python3 /home/argon/Projects/Krishan/ATPC/scripts/Convert_to_CNNpt.py ${SLURM_ARRAY_TASK_ID} ${BATCH_SIZE}

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 
