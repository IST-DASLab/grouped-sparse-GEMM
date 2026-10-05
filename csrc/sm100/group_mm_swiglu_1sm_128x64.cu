// group_mm tile variant with the fused SwiGLU + FP4 epilogue: 1SM 128x64x256.
#include "swiglu_epilogue.cuh"
namespace paired_nvfp4 {
void run_group_mm_swiglu_1sm_128x64(GroupMmParams const& p, SwigluFp4Args const& a) {
  run_group_mm_swiglu_variant<SwigluGemmVariant<
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100,
      cutlass::epilogue::TmaWarpSpecialized1SmNvf4, 64>>(p, a);
}
} // namespace paired_nvfp4
