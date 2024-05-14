#!/bin/bash

#SBATCH -o ./%j.%N.log
#SBATCH -D ./
#SBATCH -J size-ms-1
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --export=NONE
#SBATCH --time=72:00:00
#SBATCH --ntasks=28
#SBATCH --ntasks-per-core=1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --mail-type=end
#SBATCH --mail-user=ge28quz@mytum.de

source /etc/profile
source /etc/profile.d/modules.sh

#source prepare.sh
module load anaconda3
source activate env1

export DFTB_COMMAND=/dss/dsshome1/lxc09/ge28quz2/.conda/envs/env1/bin/dftb+
export DFTB_PREFIX=/dss/dsshome1/lxc09/ge28quz2/dftbplus-20.2.1/recipes/slakos/mio-1-1

#python 0413_debugging.py
/dss/dsshome1/lxc09/ge28quz2/.conda/envs/env1/bin/python aml_v20230915.py
