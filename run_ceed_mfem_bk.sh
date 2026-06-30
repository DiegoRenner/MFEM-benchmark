#!/bin/bash
# CEED-faithful MFEM bake-off-kernel sweep (Phase A: Hex).
#   driver: ProfilerElmtOps_ceed  (L2 GaussLobatto/GLL nodal, q=p+2 GL, PARTIAL assembly)
#   straight cube mesh is correct for MFEM (no affine optimisation in PA).
#   BK1 = Mass, BK3 = Stiffness; backends = MFEM-native CUDA + 3 libCEED CUDA.
# Output: ~/MFEM-benchmark/build/examples/mfem_ceed_<device>_<op>_Hex.log
#         cols: size  total_dofs  order  throughput(dof/s)
set -u
cd ~/MFEM-benchmark/build/examples || exit 1
ml cuda 2>/dev/null
export LD_LIBRARY_PATH=$HOME/libCEED/lib:$LD_LIBRARY_PATH
D=./ProfilerElmtOps_ceed
backends=("cuda" "ceed-cuda:/gpu/cuda/gen" "ceed-cuda:/gpu/cuda/shared" "ceed-cuda:/gpu/cuda/ref")
rm -f mfem_ceed_*.log
for be in "${backends[@]}"; do
  for op in Mass Stiffness; do
    for s in 8 16 24 32 48 64; do
      for o in 1 2 3 4 5 6 7; do
        "$D" "$be" "$op" Hex "$s" "$o" 2>/dev/null || echo "  (fail $be $op $s $o)"
      done
    done
    echo "[$(date +%H:%M:%S)] done $be $op"
  done
done
echo "==== CEED MFEM BK SWEEP DONE [$(date +%H:%M:%S)] ===="
