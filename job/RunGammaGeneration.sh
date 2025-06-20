#!/bin/bash
#SBATCH -J GammaGeneration # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-6:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/GammaGeneration_%A_%a.out # Standard output
#SBATCH -e log/GammaGeneration_%A_%a.err # Standard error

start=`date +%s`

# Setup nexus and run
echo "Setting up environment"
source /home/argon/Projects/Krishan/venv/bin/activate
source /home/argon/Projects/Krishan/nexus/setup_cluster.sh

Enrichment=$1
Pressure=$2
Thickness=$3
MODE=$4
CubeSize=$5

echo "Enrichment mode is: ${Enrichment}"
echo "Pressure is: ${Pressure}"
echo "Thickness is: ${Thickness}"
echo "Gamma mode is: ${MODE}"
echo "Cube Size is: ${CubeSize}"

CONFIG=ATPC_${MODE}.config.mac
INIT=ATPC_${MODE}.init.mac

mkdir -p /media/argon/HardDrive_8TB/Krishan/ATPC/GammaThickness/${MODE}/${Thickness}cm/${Pressure}bar/${Enrichment}/job${SLURM_ARRAY_TASK_ID}/
cd       /media/argon/HardDrive_8TB/Krishan/ATPC/GammaThickness/${MODE}/${Thickness}cm/${Pressure}bar/${Enrichment}/job${SLURM_ARRAY_TASK_ID}/

cp /home/argon/Projects/Krishan/ATPC/config/${CONFIG} .
cp /home/argon/Projects/Krishan/ATPC/config/${INIT} .
cp /home/argon/Projects/Krishan/ATPC/scripts/SmearEnergy.py .
cp /home/argon/Projects/Krishan/ATPC/scripts/CompressEvents.py .
cp /home/argon/Projects/Krishan/ATPC/scripts/GetGammaInfo.py .

# Copy the ion generator disrupter file
if [ "$MODE" = "Bi_ion" ]; then
    cp /home/argon/Projects/Krishan/ATPC/config/Bi214.mac .
fi

SEED=$((${SLURM_ARRAY_TASK_ID} + 1))
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure ${Pressure} bar#" ${CONFIG}
sed -i "s#.*chamber_thickn.*#/Geometry/ATPC/chamber_thickn ${Thickness} cm#" ${CONFIG}
sed -i "s#.*output_file.*#/nexus/persistency/output_file ATPC_${MODE}_${Enrichment}_${Pressure}bar#" ${CONFIG}
sed -i "s#.*cube_size.*#/Geometry/ATPC/cube_size ${CubeSize} m#" ${CONFIG}

cat ${CONFIG}
cat ${INIT}

nexus ${INIT} -n 10000000


python3 CompressEvents.py ATPC_${MODE}_${Enrichment}_${Pressure}bar ATPC_${MODE}_${Enrichment}_${Pressure}bar

python3 GetGammaInfo.py  ATPC_${MODE}_${Enrichment}_${Pressure}bar ${SLURM_ARRAY_TASK_ID}

python3 SmearEnergy.py ATPC_${MODE}_${Enrichment}_${Pressure}bar

rm *.mac
rm *.py

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 