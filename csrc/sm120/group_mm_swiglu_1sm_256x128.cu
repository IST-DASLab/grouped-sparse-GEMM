// group_mm tile variant with the fused SwiGLU + FP4 epilogue: 256x128x256, grouped.
#include "swiglu_epilogue.cuh"
namespace paired_nvfp4 {
void run_group_mm_swiglu_1sm_256x128(GroupMmParams const& p, SwigluFp4Args const& a) {
  run_group_mm_swiglu_variant<SwigluGemmVariant<256, 128>>(p, a);
}
} // namespace paired_nvfp4
