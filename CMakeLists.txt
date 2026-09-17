cmake_minimum_required(VERSION 3.28)

# The reference implementation is compiled for sm_120a (RTX 5090). sm_89 (RTX 4090)
# is supported for the architecture-generic kernel set (bf16/fp8/k8v4 paths); the
# warp-specialized SM120 NVFP4 W4A4 TMA kernels are excluded from non-120 builds
# (see src/CMakeLists.txt). Validate the architecture before CUDA compiler detection
# so CMake never injects a generic host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=120a or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
# Structured JSON output (OpenAI response_format / Responses text.format) compiles the requested
# schema into a token-level automaton. OFF drops the dependency and the enforcement together; the
# serving layer then rejects every non-text format with response_format_not_supported again.
option(NINFER_ENABLE_STRUCTURED_OUTPUT "Enforce grammar-constrained JSON output" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# sm_89 (RTX 4090) builds enable the capped-shared-memory w8 kernel variants (see
# src/ops/linear/w8/w8_config.h); the 120a build keeps its default high-memory schedules.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "89")
  add_compile_definitions(NINFER_SM89=1)
endif()

if(MSVC)
  add_compile_definitions(NOMINMAX UTF8PROC_STATIC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C>:/FS>
    $<$<COMPILE_LANGUAGE:C>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/FS>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
  )
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
