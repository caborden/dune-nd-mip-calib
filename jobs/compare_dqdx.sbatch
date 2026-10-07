#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Submit from the repository root with sbatch -A ACCOUNT.
# Arguments: fsdcube root, new overlay run ID, repeated --member LABEL=DQDX_DIR.
# No explicit memory request; use the site's default.
set -Eeo pipefail
trap 'status=$?; echo "ERROR: exit $status at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR
echo "Job ${SLURM_JOB_ID:-unset}: CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}; TRES_PER_TASK=${SLURM_TRES_PER_TASK:-unset}"
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/common/software/dune/efield2_rcmanduj/
set -u
cd "$SLURM_SUBMIT_DIR"
root="${1:?Provide the fsdcube workspace root}"
run_id="${2:?Provide a new overlay run ID}"
shift 2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
step_cpus="${SLURM_CPUS_PER_TASK:?Missing allocated CPUs per task}"
env -u SLURM_TRES_PER_TASK srun --ntasks=1 --cpus-per-task="$step_cpus" --cpu-bind=cores \
    python -u -m analysis.compare_dqdx --root "$root" --run-id "$run_id" "$@"
