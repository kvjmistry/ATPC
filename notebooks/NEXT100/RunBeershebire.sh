#!/bin/bash
#SBATCH -J Bshire # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/BSHIRE_%A_%a.out # Standard output
#SBATCH -e log/BSHIRE_%A_%a.err # Standard error

start=`date +%s`

SHIFT=0
echo "The SHIFT is $SHIFT"

JOBINDEX=$((SLURM_ARRAY_TASK_ID + SHIFT))
echo "The JOBINDEX IS ${JOBINDEX}"

# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

input_file=$(sed -n "${JOBINDEX}p" /home/argon/Projects/Krishan/ATPC/notebooks/NEXT100/filelist.txt)
echo "Input File: $input_file"

python /home/argon/Projects/Krishan/ATPC/notebooks/NEXT100/beershebashire.py ${input_file}


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
