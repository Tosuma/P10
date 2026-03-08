#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err
echo "hello world $1 $2"
sleep 25
