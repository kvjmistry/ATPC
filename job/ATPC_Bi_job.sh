#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID $1 running on `whoami`@`hostname`"

SCRIPT=$3
echo "Script name is: ${SCRIPT}"

BINSIZE=$4
echo "BINSIZE is: ${BINSIZE}"

start=`date +%s`

# Setup nexus
echo "Setting Up NEXUS" 
source /software/nexus/setup_nexus.sh

# Set the configurable variables
N_EVENTS=28000
CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

echo "N_EVENTS: ${N_EVENTS}"

SEED=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
echo "The seed number is: ${SEED}" 

# Change the config in the files
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}

# Print out the config and init files
cat ${INIT}
cat ${CONFIG}

# NEXUS
echo "Running NEXUS" 
nexus -n $N_EVENTS ${INIT}
python3 ${SCRIPT} ${JOBNAME} 0 0.05 ${BINSIZE} # Just smearing
python3 ${SCRIPT} ${JOBNAME} 1 0.1  ${BINSIZE} # 0.1 % CO2
python3 ${SCRIPT} ${JOBNAME} 1 0.25 ${BINSIZE} # 0.25 % CO2
python3 ${SCRIPT} ${JOBNAME} 1 0.5  ${BINSIZE} # 0.5 % CO2
python3 ${SCRIPT} ${JOBNAME} 1 5    ${BINSIZE} # 5.0 % CO2
python3 ${SCRIPT} ${JOBNAME} 1 0.05 ${BINSIZE} # close to zero diffusion

ls -ltrh

# Remove the large file
rm ATPC_Bi.h5

echo "Taring the h5 files"
tar -cvf ATPC_Bi.tar *.h5

# Cleanup
rm *.h5
rm *.mac
# rm *.txt
rm *.py

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 