#!/bin/bash
#SBATCH -J TrackReco # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/NEXUSTRUE_%A_%a.out # Standard output
#SBATCH -e log/NEXUSTRUE_%A_%a.err # Standard error

start=`date +%s`

echo "The TASK ID IS ${SLURM_ARRAY_TASK_ID}"

# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

PRESSURE=$1
MODE=$2

mkdir -p /media/argon/HardDrive_8TB/Krishan/ATPC/NEXUSTRUE/${MODE}/${PRESSURE}bar/
cd       /media/argon/HardDrive_8TB/Krishan/ATPC/NEXUSTRUE/${MODE}/${PRESSURE}bar/

input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/ATPC/eventlists/ATPC_${MODE}_${PRESSURE}bar_nexusfiles.txt)
echo "Input File: $input_file"


python /home/argon/Projects/Krishan/ATPC/scripts/GetTrueInfo.py ${PRESSURE} ${MODE} ${input_file}


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60