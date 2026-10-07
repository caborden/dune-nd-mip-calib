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
set -Eeo pipefail
trap 'status=$?; echo "ERROR: exit $status at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR
echo "Job ${SLURM_JOB_ID:-interactive} on $(hostname); submit directory: ${SLURM_SUBMIT_DIR:-unset}"
echo "Loading conda module"
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Activating analysis environment"
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

echo "Python: $(command -v python)"
echo "Config: $config; root: $root; run: $run_id"
python -u - "$config" "$root" <<'PY'
import sys
from pathlib import Path
from analysis.io import read_json, runtime_versions
from analysis.run import resolve_target
print('Runtime:', runtime_versions(), flush=True)
directory, samples = resolve_target(Path(sys.argv[2]), read_json(sys.argv[1])['target'])
print('Target directory:', directory, flush=True)
for kind, title, path in samples:
    print(kind, path, flush=True)
PY

srun --cpu-bind=cores python -u -m analysis.run \
    --config "$config" --root "$root" --run-id "$run_id" \
    --modules all "$@"
