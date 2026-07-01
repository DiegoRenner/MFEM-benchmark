#!/bin/bash
# CEED-faithful MFEM bake-off-kernel sweep (GH200).
#   driver: ProfilerElmtOps_ceed  (L2 GaussLobatto/GLL nodal, q=p+2 GL, PARTIAL assembly)
#   straight cube mesh is correct for MFEM (no affine optimisation in PA).
#   BK1 = Mass, BK3 = Stiffness.
# Env overrides:
#   BACKENDS  default "cuda ceed-cuda:/gpu/cuda/gen ceed-cuda:/gpu/cuda/shared ceed-cuda:/gpu/cuda/ref"
#             (NOTE: native "cuda" PA is tensor-product only -> Hex; use libCEED-only for Prism/Tet)
#   SHAPES    default "Hex"     SIZES  default "8 16 24 32 48 64"     ORDERS default "1 2 3 4 5 6 7"
# Output: ~/MFEM-benchmark/build/examples/mfem_ceed_<device>_<op>_<shape>.log
#         cols: size  total_dofs  order  throughput(dof/s)
# Only logs of the swept shapes are removed on start (previous shapes preserved).
set -u
cd ~/MFEM-benchmark/build/examples || exit 1
ml cuda 2>/dev/null
export LD_LIBRARY_PATH=$HOME/libCEED/lib:$LD_LIBRARY_PATH
D=./ProfilerElmtOps_ceed
read -ra backends <<< "${BACKENDS:-cuda ceed-cuda:/gpu/cuda/gen ceed-cuda:/gpu/cuda/shared ceed-cuda:/gpu/cuda/ref}"
SHAPES=${SHAPES:-"Hex"}
SIZES=${SIZES:-"8 16 24 32 48 64"}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7"}
# NOCLEAN=1 skips cleanup -> append mode (gap-fill runs)
if [ -z "${NOCLEAN:-}" ]; then
  for sh in $SHAPES; do rm -f mfem_ceed_*_"$sh".log; done
fi
for be in "${backends[@]}"; do
  for op in Mass Stiffness; do
    for sh in $SHAPES; do
      for s in $SIZES; do
        for o in $ORDERS; do
          "$D" "$be" "$op" "$sh" "$s" "$o" 2>/dev/null || echo "  (fail $be $op $sh $s $o)"
        done
      done
    done
    echo "[$(date +%H:%M:%S)] done $be $op ($SHAPES)"
  done
done
echo "==== CEED MFEM BK SWEEP DONE [$(date +%H:%M:%S)] ===="
