#!/bin/bash
#BSUB -J train
#BSUB -o train%J.out
#BSUB -e train%J.err
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=512MB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"


cd $BLACKHOLE/Maritime_Domain_Awareness/
git checkout Niklas_super_duper_gode_ide
git pull
source dl/bin/activate
cd maritime_domain_awareness/
module load python3/3.11.13
python3 --version
python3 -m src.maritime_domain_awareness.train
 

