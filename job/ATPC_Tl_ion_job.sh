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

CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

# NEXUS
echo "Running NEXUS" 
SEED=$((${JOBID} + 1))

if [ "$MODE" == "1bar" ]; then
    # 1 bar
    N_EVENTS=69000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 1. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Tl_ion_1bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 6.182 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}

    # Get info about the gamma spectrum for events 2.3-2.6 MeV
    python3 GetGammaInfo.py ${JOBNAME}_1bar ${JOBID}

    # Smear the energy and return only ROI events
    python3 SmearEnergy.py ${JOBNAME}_1bar
    
    # Get true info about the blobs etc
    python3 GetTrueInfo.py ${JOBNAME}_1bar_Efilt 1 Gamma
    
    # <Scale Factor> <CO2Percentage> <binsize> <pressure> <JOBID>
    python3 SmearEvents.py ${JOBNAME}_1bar_Efilt 0 0.05  5 1.0 ${JOBID} # Just smearing
    python3 SmearEvents.py ${JOBNAME}_1bar_Efilt 1 0.05 20 1.0 ${JOBID} # Helium 10%
    python3 SmearEvents.py ${JOBNAME}_1bar_Efilt 1  0.1 20 1.0 ${JOBID} # 0.1 % CO2
    python3 SmearEvents.py ${JOBNAME}_1bar_Efilt 1 0.25 15 1.0 ${JOBID} # 0.25 % CO2
    python3 SmearEvents.py ${JOBNAME}_1bar_Efilt 1    5 10 1.0 ${JOBID} # 5.0 % CO2
    python3 SmearEvents.py ${JOBNAME}_1bar_Efilt 1    0 50 1.0 ${JOBID} # Pure Xe
    rm ${JOBNAME}_1bar.h5 

elif [ "$MODE" == "5bar" ]; then
    # 5 bar
    N_EVENTS=13000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 5. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Tl_ion_5bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 3.615 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}

    # Get info about the gamma spectrum for events 2.3-2.6 MeV
    python3 GetGammaInfo.py ${JOBNAME}_5bar ${JOBID}
    
    # Smear the energy and return only ROI events
    python3 SmearEnergy.py ${JOBNAME}_5bar
    
    # Get true info about the blobs etc
    python3 GetTrueInfo.py ${JOBNAME}_5bar_Efilt 1 Gamma
    
    python3 SmearEvents.py ${JOBNAME}_5bar_Efilt 0 0.05  5 5.0 ${JOBID} # Just smearing
    python3 SmearEvents.py ${JOBNAME}_5bar_Efilt 1 0.05 20 5.0 ${JOBID} # Helium 10%
    python3 SmearEvents.py ${JOBNAME}_5bar_Efilt 1    5 10 5.0 ${JOBID} # 5.0 % CO2
    rm ${JOBNAME}_5bar.h5

elif [ "$MODE" == "10bar" ]; then
    # 10 bar ------------------------------------------------------------------
    N_EVENTS=8666
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 10. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Tl_ion_10bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.870 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    
    # Get info about the gamma spectrum for events 2.3-2.6 MeV
    python3 GetGammaInfo.py ${JOBNAME}_10bar ${JOBID}
    
    # Smear the energy and return only ROI events
    python3 SmearEnergy.py ${JOBNAME}_10bar
    
    # Get true info about the blobs etc
    python3 GetTrueInfo.py ${JOBNAME}_10bar_Efilt 1 Gamma
    
    python3 SmearEvents.py ${JOBNAME}_10bar_Efilt 0 0.05  5 10.0 ${JOBID} # Just smearing
    python3 SmearEvents.py ${JOBNAME}_10bar_Efilt 1 0.05 20 10.0 ${JOBID} # Helium 10%
    python3 SmearEvents.py ${JOBNAME}_10bar_Efilt 1    5 10 10.0 ${JOBID} # 5.0 % CO2
    rm ${JOBNAME}_10bar.h5

elif [ "$MODE" == "15bar" ]; then
    # 15 bar ------------------------------------------------------------------
    N_EVENTS=7000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 15. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Tl_ion_15bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.507 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    
    # Get info about the gamma spectrum for events 2.3-2.6 MeV
    python3 GetGammaInfo.py ${JOBNAME}_15bar ${JOBID}
    
    # Smear the energy and return only ROI events
    python3 SmearEnergy.py ${JOBNAME}_15bar
    
    # Get true info about the blobs etc
    python3 GetTrueInfo.py ${JOBNAME}_15bar_Efilt 1 Gamma
    
    python3 SmearEvents.py ${JOBNAME}_15bar_Efilt 0 0.05  5 15.0 ${JOBID} # Just smearing
    python3 SmearEvents.py ${JOBNAME}_15bar_Efilt 1 0.05 20 15.0 ${JOBID} # Helium 10%
    python3 SmearEvents.py ${JOBNAME}_15bar_Efilt 1    5 10 15.0 ${JOBID} # 5.0 % CO2
    rm ${JOBNAME}_15bar.h5

elif [ "$MODE" == "25bar" ]; then
    # 25 bar ------------------------------------------------------------------
    N_EVENTS=5000
    echo "N_EVENTS: ${N_EVENTS}"
    EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
    echo "The seed number is: ${SEED}" 
    echo "The EID number is: ${EID}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
    sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure 25. bar#" ${CONFIG}
    sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_Tl_ion_25bar#" ${CONFIG}
    sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size 2.114 m#" ${CONFIG}

    cat ${INIT}
    cat ${CONFIG}

    nexus -n $N_EVENTS ${INIT}
    
    # Get info about the gamma spectrum for events 2.3-2.6 MeV
    python3 GetGammaInfo.py ${JOBNAME}_25bar ${JOBID}
    
    # Smear the energy and return only ROI events
    python3 SmearEnergy.py ${JOBNAME}_25bar
    
    # Get true info about the blobs etc
    python3 GetTrueInfo.py ${JOBNAME}_25bar_Efilt 1 Gamma
    
    python3 SmearEvents.py ${JOBNAME}_25bar_Efilt 0 0.05  5 25.0 ${JOBID} # Just smearing
    python3 SmearEvents.py ${JOBNAME}_25bar_Efilt 1 0.05 20 25.0 ${JOBID} # Helium 10%
    python3 SmearEvents.py ${JOBNAME}_25bar_Efilt 1    5 10 25.0 ${JOBID} # 5.0 % CO2
    rm ${JOBNAME}_25bar.h5
fi

ls -ltrh

echo "Taring the h5 files"
tar -cvf ATPC_Tl_ion.tar *.h5

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