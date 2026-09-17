target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/append/kv_cache_append.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/launch.cu"
)

# NVFP4 (e2m1) and k8v4 (kind::f8f6f4) append kernels are Blackwell-only.
if(NINFER_OPS_SM120)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/append/nvfp4_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/append/k8v4_launch.cu"
  )
endif()
