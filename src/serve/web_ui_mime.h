#pragma once

// One MIME mapping for the embedded web UI, shared by the serving layer and by the build-time
// embed tool (tools/ui/embed.cpp) that writes the generated asset table. Keeping both sides on
// this header is what makes the table's recorded content type and a runtime lookup agree.

#include <string_view>

namespace ninfer::serve {

[[nodiscard]] constexpr std::string_view web_ui_mime_for(std::string_view name) noexcept {
    const auto dot = name.rfind('.');
    if (dot == std::string_view::npos) { return "application/octet-stream"; }
    const std::string_view extension = name.substr(dot);
    if (extension == ".html" || extension == ".htm") { return "text/html; charset=utf-8"; }
    if (extension == ".js" || extension == ".mjs") { return "text/javascript; charset=utf-8"; }
    if (extension == ".css") { return "text/css; charset=utf-8"; }
    if (extension == ".json") { return "application/json"; }
    if (extension == ".webmanifest") { return "application/manifest+json"; }
    if (extension == ".svg") { return "image/svg+xml"; }
    if (extension == ".png") { return "image/png"; }
    if (extension == ".jpg" || extension == ".jpeg") { return "image/jpeg"; }
    if (extension == ".webp") { return "image/webp"; }
    if (extension == ".gif") { return "image/gif"; }
    if (extension == ".ico") { return "image/x-icon"; }
    if (extension == ".woff2") { return "font/woff2"; }
    if (extension == ".woff") { return "font/woff"; }
    if (extension == ".ttf") { return "font/ttf"; }
    if (extension == ".wasm") { return "application/wasm"; }
    if (extension == ".txt") { return "text/plain; charset=utf-8"; }
    if (extension == ".map") { return "application/json"; }
    return "application/octet-stream";
}

} // namespace ninfer::serve
