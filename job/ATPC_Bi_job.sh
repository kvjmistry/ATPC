#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID $1 running on `whoami`@`hostname`"

SCRIPT=$3
echo "Script name is: ${SCRIPT}"

MODE=$4
echo "Mode is: ${MODE}"

start=`date +%s`

# Setup nexus
echo "Setting Up NEXUS" 
source /software/nexus/setup_nexus.sh

# Set the configurable variables
N_EVENTS=35000
CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

echo "N_EVENTS: ${N_EVENTS}"

SEED=$((${N_EVENTS}*${JOBID} + ${N_EVENTS} + 200000))
echo "The seed number is: ${SEED}" 

# Change the config in the files
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}

# Print out the config and init files
cat ${INIT}
cat ${CONFIG}

# NEXUS
echo "Running NEXUS" 

if [ "$MODE" == "CO2" ]; then
    # 1 bar
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 1. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_1bar#" ${CONFIG}
    nexus -n $N_EVENTS ${INIT}
    # <Scale Factor> <CO2Percentage> <binsize> <pressure> <JOBID>
    python3 ${SCRIPT} ${JOBNAME}_1bar 0 0.05  5 1.0 ${JOBID} # Just smearing
    python3 ${SCRIPT} ${JOBNAME}_1bar 1 0.05  5 1.0 ${JOBID} # close to zero diffusion
    python3 ${SCRIPT} ${JOBNAME}_1bar 1  0.1 18 1.0 ${JOBID} # 0.1 % CO2
    python3 ${SCRIPT} ${JOBNAME}_1bar 1 0.25 15 1.0 ${JOBID} # 0.25 % CO2
    python3 ${SCRIPT} ${JOBNAME}_1bar 1  0.5 12 1.0 ${JOBID} # 0.5 % CO2
    python3 ${SCRIPT} ${JOBNAME}_1bar 1    5 10 1.0 ${JOBID} # 5.0 % CO2
    rm ATPC_Bi_1bar.h5
elif [ "$MODE" == "NEXUS" ]; then
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 15. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_15bar_${JOBID}#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.432 m#" ${CONFIG}
    nexus -n $N_EVENTS ${INIT}
    python3 CompressEvents.py ATPC_Bi_15bar_${JOBID}
    rm ATPC_Bi_15bar_${JOBID}.h5
else
    # 5 bar
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 5. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_5bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 3.508 m#" ${CONFIG}
    nexus -n $N_EVENTS ${INIT}
    python3 ${SCRIPT} ${JOBNAME}_5bar 0 0.05  5 5.0 ${JOBID} # Just smearing
    python3 ${SCRIPT} ${JOBNAME}_5bar 1    5 10 5.0 ${JOBID} # 5.0 % CO2

    # 10 bar
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 10. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_10bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.784 m#" ${CONFIG}
    nexus -n $N_EVENTS ${INIT}
    python3 ${SCRIPT} ${JOBNAME}_10bar 0 0.05  5 10.0 ${JOBID} # Just smearing
    python3 ${SCRIPT} ${JOBNAME}_10bar 1    5 10 10.0 ${JOBID} # 5.0 % CO2

    # 15 bar
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 15. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_15bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.432 m#" ${CONFIG}
    nexus -n $N_EVENTS ${INIT}
    python3 ${SCRIPT} ${JOBNAME}_15bar 0 0.05  5 15.0 ${JOBID} # Just smearing
    python3 ${SCRIPT} ${JOBNAME}_15bar 1    5 10 15.0 ${JOBID} # 5.0 % CO2

    # Remove the large file
    rm ATPC_Bi_5bar.h5
    rm ATPC_Bi_10bar.h5
    rm ATPC_Bi_15bar.h5
fi


ls -ltrh

echo "Taring the h5 files"
tar -cvf ATPC_Bi.tar *.h5

# Cleanup
rm *.h5
rm *.mac
# rm *.txt
rm *.dat
rm *.py

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 