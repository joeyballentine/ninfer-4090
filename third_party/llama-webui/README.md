# llama.cpp web UI

`ninfer-serve` serves the prebuilt web UI from the llama.cpp release pinned in
`cmake/NinferWebUI.cmake`. No source is vendored here: the build downloads
`llama-<tag>-ui.tar.gz`, verifies its SHA-256 and compiles the files into the binary.
`LICENSE` is llama.cpp's MIT license, which covers those files.

Upstream: https://github.com/ggml-org/llama.cpp
