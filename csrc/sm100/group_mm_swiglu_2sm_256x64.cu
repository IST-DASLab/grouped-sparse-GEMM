// group_mm tile variant with the fused SwiGLU + FP4 epilogue: 2SM 256x64x256.
#include "swiglu_epilogue.cuh"
namespace paired_nvfp4 {
void run_group_mm_swiglu_2sm_256x64(GroupMmParams const& p, SwigluFp4Args const& a) {
  run_group_mm_swiglu_variant<SwigluGemmVariant<
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmNvf4Sm100,
      cutlass::epilogue::TmaWarpSpecialized2SmNvf4, 64>>(p, a);
}
} // namespace paired_nvfp4
