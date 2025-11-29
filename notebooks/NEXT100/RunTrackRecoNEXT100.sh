#!/bin/bash
#SBATCH -J TrackReco # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/TrackReco_%A_%a.out # Standard output
#SBATCH -e log/TrackReco_%A_%a.err # Standard error

start=`date +%s`

# Setup nexus and run
echo "Setting up env"
source /home/argon/Projects/Krishan/venv/bin/activate

input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/ATPC/notebooks/NEXT100/filelist_rebinned.txt)
echo "Input File: $input_file"

# Loop over the blob radii from 40 to 150 mm in steps of 10mm
for BLOBR in $(seq 40 10 150); do
    echo "On BLOB R: ${BLOBR}"
    mkdir -p /media/argon/HardDrive_8TB/Krishan/ATPC/354015/TrackReco/blobR_${BLOBR}
    cd       /media/argon/HardDrive_8TB/Krishan/ATPC/354015/TrackReco/blobR_${BLOBR}
    
    python /home/argon/Projects/Krishan/ATPC/notebooks/NEXT100/TrackReconstructionNEXT100.py $input_file ${BLOBR} ${BLOBR} 0

done

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 