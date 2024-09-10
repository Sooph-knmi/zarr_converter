#!/bin/bash
#$ -N mtz_nofill_5km_2020-2023
#$ -b n
#$ -S /bin/bash
#$ -l h_data=20G
#$ -l h_rss=20G
#$ -l h_rt=40:00:00
#$ -q gpu-r8.q
#$ -l h=gpu-03.ppi.met.no
#$ -o ~/logs/zarrconv_nofill_2020-2023_5km_v1_log.txt
#$ -e ~/logs/zarrconv_nofill_2020-2023_5km_v1_error.txt

module use /modules/MET/rhel8/user-modules
module load singularity
singularity exec --nv -B /lustre/:/lustre/ /lustre/storeB/project/nwp/aifs/containers/mepstozarr.sif python /lustre/storeB/project/nwp/aifs/havardhh/aifs/aifs-support/zarr_converter/multi_tz.py
