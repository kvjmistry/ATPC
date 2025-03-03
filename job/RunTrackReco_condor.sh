#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID $JOBNAME running on `whoami`@`hostname`"

input_file=$3
echo "Input file name is: ${input_file}"

PRESS=$4

# Remove the "bar"
PRESS=$(echo "${PRESS}" | sed 's/[^0-9]*//g')
echo "$PRESS"
echo "Pressure is: ${PRESS}bar"

Diff=$5
echo "Diff is: ${Diff}"

start=`date +%s`

ls -ltrh

python3 TrackReconstruction.py $input_file ${PRESS} ${Diff} 0

rm TrackReconstruction_functions.py
rm ${input_file}

ls -ltrh

tar -cvf ${JOBNAME}.tar *.h5 *.pkl

rm *.h5 *.pkl


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60