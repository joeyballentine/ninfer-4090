#include "serve/web_ui.h"

#include "ninfer/types.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kIndexName = "index.html";

bool has_dot_dot_segment(std::string_view path) {
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end     = std::min(path.find('/', start), path.size());
        const std::string_view segment = path.substr(start, end - start);
        if (segment == "..") { return true; }
        start = end + 1;
    }
    return false;
}

// One server-owned prefix: matches the prefix itself, a deeper path under it, or a longer path
// only when the prefix already ends in '/'.
bool matches_prefix(std::string_view path, std::string_view prefix) {
    if (!path.starts_with(prefix)) { return false; }
    if (prefix.ends_with('/')) { return true; }
    return path.size() == prefix.size() || path[prefix.size()] == '/';
}

bool accept_lists(std::string_view accept, std::string_view type) {
    for (std::size_t index = accept.find(type); index != std::string_view::npos;
         index             = accept.find(type, index + 1)) {
        const bool left_ok  = index == 0 || accept[index - 1] == ' ' || accept[index - 1] == ',';
        const std::size_t after = index + type.size();
        const bool right_ok = after == accept.size() || accept[after] == ',' ||
                              accept[after] == ';' || accept[after] == ' ';
        if (left_ok && right_ok) { return true; }
    }
    return false;
}

void write_asset(const WebUiAsset& asset, const httplib::Request& request,
                 httplib::Response& response) {
    response.set_header("Cache-Control", std::string(web_ui_cache_control(asset.name)));
    response.set_header("ETag", std::string(asset.etag));
    if (request.get_header_value("If-None-Match") == asset.etag) {
        response.status = 304;
        return;
    }
    response.set_content(reinterpret_cast<const char*>(asset.data), asset.size,
                         std::string(asset.content_type));
}

} // namespace

#ifndef NINFER_EMBED_WEBUI
bool web_ui_has_assets() noexcept { return false; }

std::span<const WebUiAsset> web_ui_assets() noexcept { return {}; }
#endif

const WebUiAsset* web_ui_find_asset(std::string_view name) noexcept {
    if (name.empty()) { return nullptr; }
    for (const WebUiAsset& asset : web_ui_assets()) {
        if (asset.name == name) { return &asset; }
    }
    return nullptr;
}

std::string_view web_ui_asset_name(std::string_view request_path) noexcept {
    if (request_path.empty() || request_path.front() != '/') { return {}; }
    if (request_path == "/" || request_path == "/index.html") { return kIndexName; }
    const std::string_view name = request_path.substr(1);
    if (name.empty() || has_dot_dot_segment(name)) { return {}; }
    return name;
}

std::string_view web_ui_cache_control(std::string_view name) noexcept {
    // SvelteKit writes the content hash into every _app/immutable/** file name, so a stale copy is
    // impossible: a changed file is a changed URL.
    if (name.starts_with("_app/immutable/")) { return "public, max-age=31536000, immutable"; }
    // The entry document, the service worker and the PWA descriptors decide which immutable URLs
    // the browser loads, so they must be revalidated on every load.
    if (name == kIndexName || name == "sw.js" || name == "build.json" ||
        name == "_app/version.json" || name.ends_with(".webmanifest")) {
        return "no-cache";
    }
    return "public, max-age=3600";
}

bool web_ui_is_api_path(std::string_view path) noexcept {
    return matches_prefix(path, "/v1") || matches_prefix(path, "/health") ||
           matches_prefix(path, "/metrics") || matches_prefix(path, "/slots") ||
           matches_prefix(path, "/props");
}

bool web_ui_accepts_html(std::string_view accept_header) noexcept {
    if (accept_header.empty()) { return false; }
    return accept_lists(accept_header, "text/html") ||
           accept_lists(accept_header, "application/xhtml+xml") ||
           accept_lists(accept_header, "*/*") || accept_lists(accept_header, "text/*");
}

bool web_ui_should_serve_index(std::string_view method, std::string_view path,
                               std::string_view accept_header) noexcept {
    if (!web_ui_has_assets()) { return false; }
    if (method != "GET" && method != "HEAD") { return false; }
    if (web_ui_is_api_path(path)) { return false; }
    if (!web_ui_accepts_html(accept_header)) { return false; }
    // A real embedded file is served as itself; only unmatched routes get the SPA document.
    return web_ui_find_asset(web_ui_asset_name(path)) == nullptr;
}

bool web_ui_is_public_request(std::string_view method, std::string_view path) noexcept {
    if (!web_ui_has_assets()) { return false; }
    if (method != "GET" && method != "HEAD") { return false; }
    return !web_ui_is_api_path(path);
}

std::string render_web_ui_props(const WebUiPropsInput& input) {
    static const ServeOptions kDefaults{};
    const ServeOptions& options = input.options != nullptr ? *input.options : kDefaults;

    Json params;
    if (input.sampling_defaults != nullptr) {
        // The non-thinking preset is what a plain chat turn resolves to; the UI only uses these to
        // label a field as "server default".
        const ninfer::SamplingPreset& preset = input.sampling_defaults->non_thinking;
        params["temperature"]                = preset.temperature;
        params["top_k"]                      = preset.top_k;
        params["top_p"]                      = preset.top_p;
        params["min_p"]                      = preset.min_p;
        params["presence_penalty"]           = preset.presence_penalty;
        params["frequency_penalty"]          = preset.frequency_penalty;
    }
    params["max_tokens"] = options.default_max_tokens;

    // NInfer applies the model artifact's own chat template inside the Engine and never exposes the
    // template text. The UI reads this field for two things: to show where the template came from,
    // and to decide whether to offer its thinking controls, which it does by looking for the
    // enable_thinking kwarg. Reporting the kwarg the server actually honours is the accurate answer
    // to both.
    std::string chat_template =
        options.chat_template_path.empty()
            ? std::string("{# NInfer applies the model artifact's chat template in-engine. #}")
            : "{# NInfer applies " + options.chat_template_path.generic_string() +
                  " in-engine. #}";
    if (options.enable_thinking.value_or(true)) {
        chat_template += "\n{%- if enable_thinking %}{%- endif %}";
    }

    const Json props = {
        {"model_path", options.artifact_path},
        {"model_alias", input.model_id},
        {"build_info", input.build_info},
        {"total_slots", options.max_concurrency},
        {"n_ctx", options.max_context},
        {"chat_template", std::move(chat_template)},
        {"modalities",
         {{"vision", options.enable_vision}, {"audio", false}, {"video", false}}},
        {"default_generation_settings",
         {{"n_ctx", options.max_context}, {"params", std::move(params)}}},
        {"endpoint_props", true},
        {"endpoint_slots", true},
        {"endpoint_metrics", true},
    };
    return props.dump();
}

void register_web_ui_routes(httplib::Server& server, std::function<std::string()> render_props) {
    if (!web_ui_has_assets()) { return; }

    server.Get("/props", [render_props = std::move(render_props)](const httplib::Request&,
                                                                  httplib::Response& response) {
        response.set_content(render_props(), "application/json");
    });

    // One catch-all keeps the table the single routing authority: any embedded path is served as
    // itself, an HTML-accepting miss outside the server's own prefixes gets the SPA document, and
    // everything else falls through to the normal 404 contract.
    server.Get(R"(.*)", [](const httplib::Request& request, httplib::Response& response) {
        const WebUiAsset* asset = web_ui_find_asset(web_ui_asset_name(request.path));
        if (asset != nullptr) {
            write_asset(*asset, request, response);
            return;
        }
        if (web_ui_should_serve_index(request.method, request.path,
                                      request.get_header_value("Accept"))) {
            const WebUiAsset* index = web_ui_find_asset(kIndexName);
            if (index != nullptr) {
                write_asset(*index, request, response);
                return;
            }
        }
        // Empty body: the server's error handler renders the documented 404 for API paths and
        // cpp-httplib's default for anything else, exactly as an unrouted path does today.
        response.status = 404;
    });
}

} // namespace ninfer::serve
