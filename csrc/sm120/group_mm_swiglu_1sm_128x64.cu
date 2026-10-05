// group_mm tile variant with the fused SwiGLU + FP4 epilogue: 128x64x256, grouped.
#include "swiglu_epilogue.cuh"
namespace paired_nvfp4 {
void run_group_mm_swiglu_1sm_128x64(GroupMmParams const& p, SwigluFp4Args const& a) {
  run_group_mm_swiglu_variant<SwigluGemmVariant<128, 64>>(p, a);
}
} // namespace paired_nvfp4
