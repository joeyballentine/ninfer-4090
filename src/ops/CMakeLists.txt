# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)

# Blackwell-only kernel policy. NVFP4 W4A4 needs TMA and the block-scaled FP4 MMA, the k8v4 KV
# format needs kind::f8f6f4, and the warp-specialized NVFP4 attention needs setmaxnreg; none of
# these exist on sm_89. Each family excludes its own Blackwell sources on a non-120 build and the
# two stub units below keep the dispatch layer linkable with a clear runtime error instead.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "89")
  set(NINFER_OPS_SM89 ON)
else()
  set(NINFER_OPS_SM89 OFF)
endif()

if(CMAKE_CUDA_ARCHITECTURES MATCHES "^120")
  set(NINFER_OPS_SM120 ON)
else()
  set(NINFER_OPS_SM120 OFF)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4_stubs.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/sm120_kv_stubs.cpp"
  )
endif()

if(NINFER_OPS_SM120)
  # Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
  # producer to consumer warpgroups. RDC makes ptxas discard that contract. These
  # self-contained kernels therefore use a non-RDC archive and host launchers.
  add_library(ninfer_nvfp4_non_rdc STATIC)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
  target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)
  target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)
else()
  target_link_libraries(ninfer_ops PUBLIC ninfer_core)
endif()

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")
