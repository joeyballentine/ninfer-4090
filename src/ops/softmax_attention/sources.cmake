target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# NVFP4 and k8v4 cache reads decode e2m1 / contract through kind::f8f6f4, and the NVFP4 prompt
# kernel is warp-specialized with setmaxnreg. All three are Blackwell-only.
if(NINFER_OPS_SM120)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  )
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_non_rdc.cu"
  )
endif()
