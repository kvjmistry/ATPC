#!/bin/bash
#SBATCH -J TrackReco # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o TrackReco_%A_%a.out # Standard output
#SBATCH -e TrackReco_%A_%a.err # Standard error

start=`date +%s`

# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

SAMPLE=0nubb

mkdir -p /media/argon/HardDrive_8TB/Krishan/ATPC/RECO/${SAMPLE}/
cd       /media/argon/HardDrive_8TB/Krishan/ATPC/RECO/${SAMPLE}/
cp /home/argon/Projects/Krishan/ATPC/notebooks/TrackReconstruction_functions.py .


input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/ATPC/eventlists/${SAMPLE}_files.txt)
echo "Input File: $input_file"


python /home/argon/Projects/Krishan/ATPC/notebooks/TrackReconstruction.py $input_file 0 ${SLURM_ARRAY_TASK_ID}


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60