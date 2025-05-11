#!/bin/bash

echo "Starting Job" 

JOBID=$1
# JOBID=$((JOBID + 3000))
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID ${JOBNAME} running on `whoami`@`hostname`"

SCRIPT=$3
echo "Script name is: ${SCRIPT}"

MODE=$4
echo "Mode is: ${MODE}"

start=`date +%s`

# Setup nexus
echo "Setting Up NEXUS" 
source /software/nexus/setup_nexus.sh

# Set the configurable variables

CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

# NEXUS
echo "Running NEXUS" 

N_EVENTS=1000000
echo "N_EVENTS: ${N_EVENTS}"
SEED=$((${N_EVENTS}*${JOBID} + ${N_EVENTS} + 600000))
echo "The seed number is: ${SEED}" 
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}
sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 15. bar#" ${CONFIG}
sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_ion#" ${CONFIG}
sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.432 m#" ${CONFIG}

cat ${INIT}
cat ${CONFIG}

nexus -n $N_EVENTS ${INIT}
python3 ${SCRIPT} ${JOBNAME} ${JOBID}

ls -ltrh

# Cleanup
rm *.mac
rm *.txt
rm *.dat
rm *.py

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 