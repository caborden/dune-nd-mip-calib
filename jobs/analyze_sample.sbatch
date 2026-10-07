#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Submit from the repository root with sbatch -A ACCOUNT.
# Arguments: config path, fsdcube root, run ID, optional analysis.run flags.
set -eo pipefail
module load conda
conda activate /global/common/software/dune/efield2_rcmanduj/
set -u

cd "$SLURM_SUBMIT_DIR"
config="${1:?Provide an analysis config path}"
root="${2:?Provide the fsdcube workspace root}"
run_id="${3:?Provide a run ID}"
shift 3

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

srun --cpu-bind=cores python -u -m analysis.run \
    --config "$config" --root "$root" --run-id "$run_id" \
    --modules all "$@"
