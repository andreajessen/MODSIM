#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --mem=64000
#SBATCH --job-name="yolov8"
#SBATCH --output=test-srun2.out
#SBATCH --mail-user=solveijm@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Anaconda3/2022.05
pip install ultralytics
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_30e_1920imgsz/weights/best.pt source= '/cluster/home/solveijm/hurtigruten/test.txt' imgsz=1920 name=test_30e_1920imgsz save=True save_conf=True save_txt=True