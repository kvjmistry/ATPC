#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID $JOBNAME running on `whoami`@`hostname`"

input_file=$3
echo "Input file name is: ${input_file}"

start=`date +%s`

ls -ltrh

python TrackReconstruction.py $input_file 0 ${JOBID}

rm TrackReconstruction_functions.py
rm ${input_file}

ls -ltrh

tar -cvf ${JOBNAME}.tar *.h5 *.pkl


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60