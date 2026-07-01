#!/bin/bash
#SBATCH --job-name=run_simulation
#SBATCH --partition=cpu-long,cpu-unlimited,cpu-medium
#SBATCH --mail-user=s343310@studenti.polito.it
#SBATCH --mail-type=NONE
#SBATCH --output=/beegfs/users/mchiodo/Simulator/slurm_logs/seed050102/output_%a.txt
#SBATCH --error=/beegfs/users/mchiodo/Simulator/slurm_logs/seed050102/error_%a.txt
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=1,2,3,4,11,12,13,14,21,22,23,24,31,32,33,34

# ============================================================
# UNICO PARAMETRO DA MODIFICARE AD OGNI LANCIO
# ============================================================
export SIM_SEED=50102

# ============================================================

# Vai nella cartella del progetto
cd /beegfs/users/mchiodo/Simulator

# Licenza WLS Academic Gurobi
export GRB_LICENSE_FILE=/beegfs/users/mchiodo/.gurobi.lic


module load Python/3.11.5-GCCcore-13.2.0

pip install --user -q -r requirements.txt

echo "Avvio esperimento $SLURM_ARRAY_TASK_ID con seed $SIM_SEED su $(hostname) alle $(date)"

python3 run_simulation.py

echo "Fine esperimento $SLURM_ARRAY_TASK_ID alle $(date)"