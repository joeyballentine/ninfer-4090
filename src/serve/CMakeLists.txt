add_library(ninfer_serve STATIC
  http_transport.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  operational_log.cpp
  request_log.cpp
  http_server.cpp
  web_ui.cpp
  openai_responses_http.cpp
  serve_metrics.cpp
  serve_options.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging CUDA::cudart)

# The web UI is embedded as a generated translation unit produced by cmake/NinferWebUI.cmake. The
# definition is PUBLIC so tests can tell an embedded build from a stub build.
if(NINFER_EMBED_WEBUI)
  target_sources(ninfer_serve PRIVATE "${NINFER_WEBUI_GENERATED_CPP}")
  add_dependencies(ninfer_serve ninfer_webui_assets)
  target_compile_definitions(ninfer_serve PUBLIC NINFER_EMBED_WEBUI=1)
endif()
