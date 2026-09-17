#pragma once

// Embedded llama.cpp web UI: the generated asset table plus the routing, caching and `/props`
// policy that serves it. Everything here compiles unconditionally; only the table itself comes
// from generated code, and with NINFER_EMBED_WEBUI off the table is empty and no route is
// registered.

#include "serve/serve_options.h"
#include "serve/web_ui_mime.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace ninfer {
struct ModelSamplingDefaults;
}

namespace httplib {
class Server;
}

namespace ninfer::serve {

struct WebUiAsset {
    std::string_view name;      // serve path relative to the UI root, no leading slash
    const unsigned char* data;  // not NUL-terminated at `size`; the literal's NUL follows it
    std::size_t size;
    std::string_view etag;         // quoted strong validator
    std::string_view content_type; // web_ui_mime_for(name), recorded when the table was generated
};

// Defined by the generated translation unit when NINFER_EMBED_WEBUI is on, and by web_ui.cpp
// otherwise.
[[nodiscard]] bool web_ui_has_assets() noexcept;
[[nodiscard]] std::span<const WebUiAsset> web_ui_assets() noexcept;

[[nodiscard]] const WebUiAsset* web_ui_find_asset(std::string_view name) noexcept;

// `/` and `/index.html` both resolve to "index.html"; any other path loses its leading slash.
// Returns an empty view for a path that must never reach the table (empty, absolute-escaping,
// or containing a `..` segment).
[[nodiscard]] std::string_view web_ui_asset_name(std::string_view request_path) noexcept;

// Cache-Control for an asset name. Content-hashed `_app/immutable/**` is immutable for a year;
// the entry document, service worker, manifest and version files must be revalidated.
[[nodiscard]] std::string_view web_ui_cache_control(std::string_view name) noexcept;

// Paths the server owns: never answered with the SPA document, and never served without the API
// key when one is configured.
[[nodiscard]] bool web_ui_is_api_path(std::string_view path) noexcept;

// SPA fallback: an unmatched GET that asks for HTML and is not an API path gets index.html so the
// UI's client-side routes (for example /chat/<id>) survive a reload.
[[nodiscard]] bool web_ui_accepts_html(std::string_view accept_header) noexcept;
[[nodiscard]] bool web_ui_should_serve_index(std::string_view method, std::string_view path,
                                             std::string_view accept_header) noexcept;

// True when the request may be answered without the configured API key: the static documents are
// public, everything else (including /props) stays authenticated.
[[nodiscard]] bool web_ui_is_public_request(std::string_view method,
                                            std::string_view path) noexcept;

// `GET /props` payload. UI-support only; see docs/serving.md.
struct WebUiPropsInput {
    const ServeOptions* options = nullptr;
    std::string model_id;
    std::string build_info;
    const ninfer::ModelSamplingDefaults* sampling_defaults = nullptr;
};

[[nodiscard]] std::string render_web_ui_props(const WebUiPropsInput& input);

// Registers `GET /`, `GET /props`, the per-asset routes and the SPA fallback. Must run after the
// API routes so cpp-httplib matches them first, and is a no-op when nothing is embedded.
void register_web_ui_routes(httplib::Server& server, std::function<std::string()> render_props);

} // namespace ninfer::serve
