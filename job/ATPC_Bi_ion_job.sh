#!/bin/bash

echo "Starting Job" 

JOBID=$1
SHIFT=0
JOBID=$((JOBID + SHIFT))
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
SEED=$((${JOBID} + 1))

if [ "$MODE" == "1bar" ]; then
    # 1 bar
    N_EVENTS=5000000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 1. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_ion_1bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 6 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    python3 CompressEvents.py ${JOBNAME}_1bar ${JOBNAME}_1bar # also filters 60 events
    python3 ${SCRIPT} ${JOBNAME}_1bar ${JOBID}
    mv ${JOBNAME}_1bar.h5 ${JOBNAME}_1bar_nexus_${JOBID}.h5


elif [ "$MODE" == "5bar" ]; then
    # 5 bar
    N_EVENTS=1000000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 5. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_ion_5bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 3.508 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    python3 CompressEvents.py ${JOBNAME}_5bar ${JOBNAME}_5bar # also filters 60 events
    python3 ${SCRIPT} ${JOBNAME}_5bar ${JOBID}
    mv ${JOBNAME}_5bar.h5 ${JOBNAME}_5bar_nexus_${JOBID}.h5

elif [ "$MODE" == "10bar" ]; then
    # 10 bar ------------------------------------------------------------------
    N_EVENTS=1000000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 10. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_ion_10bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.784 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    python3 CompressEvents.py ${JOBNAME}_10bar ${JOBNAME}_10bar # also filters 60 events
    python3 ${SCRIPT} ${JOBNAME}_10bar ${JOBID}
    mv ${JOBNAME}_10bar.h5 ${JOBNAME}_10bar_nexus_${JOBID}.h5

elif [ "$MODE" == "15bar" ]; then
    # 15 bar ------------------------------------------------------------------
    N_EVENTS=1000000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 15. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_ion_15bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.432 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    python3 CompressEvents.py ${JOBNAME}_15bar ${JOBNAME}_15bar # also filters 60 events
    python3 ${SCRIPT} ${JOBNAME}_15bar ${JOBID}
    mv ${JOBNAME}_15bar.h5 ${JOBNAME}_15bar_nexus_${JOBID}.h5

elif [ "$MODE" == "25bar" ]; then
    # 25 bar ------------------------------------------------------------------
    N_EVENTS=1000000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 25. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Bi_ion_25bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.052 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    python3 CompressEvents.py ${JOBNAME}_25bar ${JOBNAME}_25bar # also filters 60 events
    python3 ${SCRIPT} ${JOBNAME}_25bar ${JOBID}
    mv ${JOBNAME}_25bar.h5 ${JOBNAME}_25bar_nexus_${JOBID}.h5
fi

ls -ltrh

echo "Taring the h5 files"
tar -cvf ATPC_Bi_ion.tar *.h5

# Cleanup
rm *.h5
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