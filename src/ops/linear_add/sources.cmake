target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_linear_add_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_linear_add_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_linear_add_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_linear_add_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_add_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_add_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_add_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_add_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_add_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_add_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_linear_add_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4/q4_linear_add.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/q5_linear_add_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/q5_linear_add_gemm_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/q5_linear_add_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5/q5_linear_add_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_add_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_add_gemm_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_add_gemm_splitk.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_add_gemm_capacity.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_add_gemm_grouped.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_linear_add_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/linear_add.cpp"
)

# NVFP4 W4A4 contracts through the Blackwell block-scaled FP4 MMA.
if(NINFER_OPS_SM120)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_linear_add_w4a4.cu"
  )
endif()
