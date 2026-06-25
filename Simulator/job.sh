#!/bin/bash
#SBATCH --job-name=mchiodo_tesi
#SBATCH --partition=cpu-long
#SBATCH --mail-user=s343310@studenti.polito.it
#SBATCH --mail-type=ALL
#SBATCH --output=/beegfs/users/mchiodo/Simulator/slurm_logs/output_%a.txt
#SBATCH --error=/beegfs/users/mchiodo/Simulator/slurm_logs/error_%a.txt
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=1,2,3,4,11,12,13,14,21,22,23,24,31,32,33,34

# Vai nella cartella del progetto
cd /beegfs/users/mchiodo/Simulator

# Crea cartella log se non esiste
mkdir -p slurm_logs
module load Python/3.11.5-GCCcore-13.2.0

# Installa pacchetti solo se necessario
pip install --user -q -r requirements.txt

echo "Avvio esperimento $SLURM_ARRAY_TASK_ID su $(hostname) alle $(date)"

python3 run_simulation.py

echo "Fine esperimento $SLURM_ARRAY_TASK_ID alle $(date)"