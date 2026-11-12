#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

# Arguments: data|mc, merged tracks file, display label, new output directory
set -eo pipefail
kind=${1:?Provide data or mc}
tracks=${2:?Provide the merged track-selection file}
label=${3:?Provide a display label}
out=${4:?Provide a new sample output directory}
case "$kind" in data|mc) ;; *) echo "Expected data or mc" >&2; exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/common/software/dune/efield2_rcmanduj/
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

python -u analyze_track_dqdx.py "--$kind" "$tracks" --output-dir "$out"
python -u -m analysis.dqdx_angle_grid --segments "$out/segments.hdf5" \
    --sample "$kind" --label "$label" --vertical xy --stats-threshold 200 \
    --output "$out/dqdx_yz_by_xy_fit.pdf"
