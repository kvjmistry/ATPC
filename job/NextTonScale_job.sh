#!/bin/bash

echo "Starting Job" 

JOBID=$1
SHIFT=0
JOBID=$((JOBID + SHIFT))
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID ${JOBNAME} running on `whoami`@`hostname`"

MODE=$3
echo "Mode is: ${MODE}"

start=`date +%s`

# Setup nexus
echo "Setting Up NEXUS" 
source /software/nexus/setup_nexus.sh

# Set the configurable variables
CONFIG=${JOBNAME}_${MODE}.config.mac
INIT=${JOBNAME}_${MODE}.init.mac

# NEXUS
echo "Running NEXUS" 
SEED=$((${JOBID} + 1))

if [ "$MODE" == "0nubb" ]; then
    N_EVENTS=625

elif [ "$MODE" == "Bi_ion" ]; then
    N_EVENTS=5e6

elif [ "$MODE" == "Tl_ion" ]; then
    N_EVENTS=5e6
fi

echo "N_EVENTS: ${N_EVENTS}"
EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
echo "The seed number is: ${SEED}" 
echo "The EID number is: ${EID}" 
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}

cat ${INIT}
cat ${CONFIG}

nexus -n $N_EVENTS ${INIT}

# Smear the energy, voxelize and return only window events
python3 SmearEnergyNEXT1t.py ${JOBNAME}_${MODE}

mv ${JOBNAME}_${MODE}_Efilt.h5 ${JOBNAME}_${MODE}_nexus_${JOBID}.h5

python3 TrackReconstruction.py ${JOBNAME}_${MODE}_nexus_${JOBID}.h5 15 "next1t" 0

rm ${JOBNAME}_${MODE}.h5

ls -ltrh

echo "Taring the h5 files"
tar -cvf ${JOBNAME}.tar *.h5 *.pkl

# Cleanup
rm *.h5
rm *.mac
rm *.txt
rm *.dat
rm *.py
rm *.pkl

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 