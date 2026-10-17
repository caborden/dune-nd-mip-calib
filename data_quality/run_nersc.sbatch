#!/bin/bash
#SBATCH --job-name=fsdcube-dq
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

# Submit from the repository root; pass your allocation with sbatch -A ACCOUNT.
# Arguments: config JSON, NEW output directory. No GPU or track skim required.
set -Eeo pipefail
trap 'status=$?; echo "ERROR: exit $status at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/common/software/dune/efield2_rcmanduj/
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the dune-nd-mip-calib repository root}"
config="${1:?Provide the config JSON}"
output="${2:?Provide a NEW output directory}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fsdcube-dq-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
echo "Job $SLURM_JOB_ID on $(hostname); config: $config; output: $output"
step_cpus="${SLURM_CPUS_PER_TASK:?Missing allocated CPUs per task}"
env -u SLURM_TRES_PER_TASK srun --ntasks=1 --cpus-per-task="$step_cpus" --cpu-bind=cores \
    python -u -m data_quality.run --config "$config" --output-dir "$output"
