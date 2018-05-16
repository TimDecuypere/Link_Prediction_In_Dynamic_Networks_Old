#!/bin/bash -l
#PBS -l nodes=1:ppn=4
#PBS -l pmem=20gb
#PBS -l walltime=24:00:00
#PBS -N python_job_node2vec
#PBS -M sam.dewinter@student.kuleuven.be
#PBS -o stdout.$PBS_JOBID
#PBS -e stderr.$PBS_JOBID

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate py27
cd $PBS_O_WORKDIR 

python Stackexchange.py 