find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(MSVC)
  # Windows has no pkg-config; the FFmpeg shared build is unpacked into the repository and
  # consumed through the same imported target the rest of the tree links.
  add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
  target_include_directories(PkgConfig::FFMPEG INTERFACE "${PROJECT_SOURCE_DIR}/ffmpeg/include")
  target_link_directories(PkgConfig::FFMPEG INTERFACE "${PROJECT_SOURCE_DIR}/ffmpeg/lib")
  target_link_libraries(PkgConfig::FFMPEG INTERFACE avformat avcodec avutil swscale swresample)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
endif()

# XGrammar owns the tokenizer-aware automaton behind structured JSON output; NInfer owns the
# per-request matcher state, the mask upload and the sampling kernels that consume the mask. Its
# sources are compiled into a local static library rather than adding its own project, so its
# project-wide compiler flags never reach the NInfer CUDA build. This is the single configure-time
# download in the tree and it is opt-out: NINFER_ENABLE_STRUCTURED_OUTPUT=OFF configures without
# network access and restores the serving-layer rejection of every non-text response format.
if(NINFER_ENABLE_STRUCTURED_OUTPUT)
  include(FetchContent)
  FetchContent_Declare(
    xgrammar_source
    GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
    # v0.2.5.post1 == caeaf129e575cfa2a65e31e71ba5f74312a1eabb
    GIT_TAG v0.2.5.post1
    GIT_SHALLOW TRUE
    GIT_SUBMODULES 3rdparty/dlpack
    # `include/` carries no CMakeLists.txt, so the content is populated without configuring
    # XGrammar's own project.
    SOURCE_SUBDIR include)
  FetchContent_MakeAvailable(xgrammar_source)
  add_library(ninfer_xgrammar STATIC
    "${xgrammar_source_SOURCE_DIR}/cpp/compiled_grammar.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/config.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/earley_parser.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/fsm.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/fsm_builder.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar_builder.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar_compiler.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar_functor.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar_matcher.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar_parser.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/grammar_printer.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/json_schema_converter.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/json_schema_converter_ext.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/lark_converter.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/regex_converter.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/structural_tag.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/testing.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/tokenizer_info.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/support/logging.cc"
    "${xgrammar_source_SOURCE_DIR}/cpp/support/recursion_guard.cc")
  target_compile_features(ninfer_xgrammar PUBLIC cxx_std_20)
  target_compile_definitions(ninfer_xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
  target_include_directories(ninfer_xgrammar SYSTEM PUBLIC
    "${xgrammar_source_SOURCE_DIR}/include"
    "${xgrammar_source_SOURCE_DIR}/3rdparty/picojson"
    "${xgrammar_source_SOURCE_DIR}/3rdparty/dlpack/include")
  set_target_properties(ninfer_xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(ninfer::xgrammar ALIAS ninfer_xgrammar)
  # Every consumer compiles the enforcement paths against the same switch.
  add_compile_definitions(NINFER_STRUCTURED_OUTPUT=1)
endif()

# Repository-pinned header dependencies. Nothing else downloads at configure time.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
