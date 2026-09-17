# llama.cpp web UI: fetch the pinned release at configure time, embed it into ninfer_serve at
# build time. Controlled by NINFER_EMBED_WEBUI (declared in the top-level CMakeLists.txt).
#
# Upstream NInfer policy avoids configure-time downloads; this fork ships the UI, so the option
# defaults ON here. An offline build either turns it OFF or points NINFER_WEBUI_TARBALL at a
# previously downloaded copy of the same pinned archive.

# The pin. One tag, one digest, nowhere else in the tree.
set(NINFER_WEBUI_TAG "b10655")
set(NINFER_WEBUI_SHA256 "2b6532cf35827bc5306cfb8447b46c3fbe3a2161177610bcb547f4da0d532862")

set(NINFER_WEBUI_TARBALL "" CACHE FILEPATH
  "Local llama-<tag>-ui.tar.gz to embed instead of downloading (offline builds)")

set(NINFER_WEBUI_ARCHIVE_NAME "llama-${NINFER_WEBUI_TAG}-ui.tar.gz")
set(NINFER_WEBUI_ASSET_DIR "${CMAKE_BINARY_DIR}/_deps/webui-${NINFER_WEBUI_TAG}")
set(NINFER_WEBUI_STAMP "${NINFER_WEBUI_ASSET_DIR}.stamp")
set(NINFER_WEBUI_GENERATED_CPP "${CMAKE_BINARY_DIR}/generated/serve/web_ui_assets.cpp")

function(_ninfer_webui_require_sha256 path)
  file(SHA256 "${path}" _actual)
  if(NOT _actual STREQUAL NINFER_WEBUI_SHA256)
    message(FATAL_ERROR
      "NInfer web UI: SHA-256 mismatch for ${path}\n"
      "  expected ${NINFER_WEBUI_SHA256}\n"
      "  actual   ${_actual}\n"
      "The archive must be the pinned llama.cpp ${NINFER_WEBUI_TAG} UI release. Re-download it, "
      "or configure with -DNINFER_EMBED_WEBUI=OFF to build without the web UI.")
  endif()
endfunction()

# Acquire and extract the pinned archive unless a matching extraction is already on disk.
if(NOT EXISTS "${NINFER_WEBUI_STAMP}" OR NOT EXISTS "${NINFER_WEBUI_ASSET_DIR}")
  set(_archive "${CMAKE_BINARY_DIR}/_deps/${NINFER_WEBUI_ARCHIVE_NAME}")
  if(NINFER_WEBUI_TARBALL)
    if(NOT EXISTS "${NINFER_WEBUI_TARBALL}")
      message(FATAL_ERROR "NInfer web UI: NINFER_WEBUI_TARBALL=${NINFER_WEBUI_TARBALL} not found")
    endif()
    message(STATUS "NInfer web UI: using local archive ${NINFER_WEBUI_TARBALL}")
    set(_archive "${NINFER_WEBUI_TARBALL}")
  else()
    set(_url "https://github.com/ggml-org/llama.cpp/releases/download/${NINFER_WEBUI_TAG}/${NINFER_WEBUI_ARCHIVE_NAME}")
    message(STATUS "NInfer web UI: downloading ${_url}")
    file(DOWNLOAD "${_url}" "${_archive}" STATUS _status TIMEOUT 300 TLS_VERIFY ON)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
      list(GET _status 1 _message)
      file(REMOVE "${_archive}")
      message(FATAL_ERROR
        "NInfer web UI: download of ${_url} failed (${_message}). Configure with "
        "-DNINFER_EMBED_WEBUI=OFF, or with -DNINFER_WEBUI_TARBALL=/path/${NINFER_WEBUI_ARCHIVE_NAME}.")
    endif()
  endif()

  _ninfer_webui_require_sha256("${_archive}")

  file(REMOVE_RECURSE "${NINFER_WEBUI_ASSET_DIR}")
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${NINFER_WEBUI_ASSET_DIR}")
  # The stamp carries the digest, so bumping the pin re-extracts rather than silently reusing the
  # previous release's files.
  file(WRITE "${NINFER_WEBUI_STAMP}" "${NINFER_WEBUI_SHA256}\n")
  message(STATUS "NInfer web UI: extracted ${NINFER_WEBUI_TAG} into ${NINFER_WEBUI_ASSET_DIR}")
endif()

# Host tool: plain C++17, no NInfer libraries and no CUDA, so it runs on the build machine.
add_executable(ninfer_webui_embed "${PROJECT_SOURCE_DIR}/tools/ui/embed.cpp")
target_compile_features(ninfer_webui_embed PRIVATE cxx_std_20)
target_include_directories(ninfer_webui_embed PRIVATE "${PROJECT_SOURCE_DIR}/src")
set_target_properties(ninfer_webui_embed PROPERTIES CUDA_ARCHITECTURES "")

add_custom_command(
  OUTPUT "${NINFER_WEBUI_GENERATED_CPP}"
  COMMAND ninfer_webui_embed "${NINFER_WEBUI_ASSET_DIR}" "${NINFER_WEBUI_GENERATED_CPP}"
  DEPENDS ninfer_webui_embed "${NINFER_WEBUI_STAMP}"
  COMMENT "Embedding llama.cpp web UI ${NINFER_WEBUI_TAG} into ninfer_serve"
  VERBATIM)

# The generated source is consumed from src/serve. The custom command only produces a rule in the
# directory that owns a target depending on it, so the utility target lives here with it.
add_custom_target(ninfer_webui_assets DEPENDS "${NINFER_WEBUI_GENERATED_CPP}")
set_source_files_properties("${NINFER_WEBUI_GENERATED_CPP}" PROPERTIES GENERATED TRUE)
