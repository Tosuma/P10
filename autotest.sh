#!/bin/bash

run_worker() {
	worker_id=$1
	for i in {1..10}; do
		echo "Starting iteration $i" 
		job_id=$(sbatch test.sh | grep -o '[0-9]\+')

		echo "Submitted Job with ID: $job_id"

		#while squeue --me | grep -q "$job_id"; do
		echo "Job $job_id still running... sleeping 60 seconds"
		sleep 2
		

		echo "Worker: $worker_id Job $job_id finished"
	done
}

run_worker 1 &
run_worker 2 &
run_worker 3 &

wait

echo "DONE!"
