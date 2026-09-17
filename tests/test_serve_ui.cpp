// Web UI serving policy: the generated asset table, MIME and cache decisions, the SPA fallback
// rule and the /props payload. Skips (77) when the build has no embedded UI.

#include "serve/web_ui.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>
#include <string_view>

namespace {

using namespace ninfer::serve;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

void check_mime_table() {
    check(web_ui_mime_for("index.html") == "text/html; charset=utf-8", "mime html");
    check(web_ui_mime_for("_app/immutable/bundle.CRRRM4bk.js") == "text/javascript; charset=utf-8",
          "mime js");
    check(web_ui_mime_for("_app/immutable/assets/bundle.3BIcW1B_.css") ==
              "text/css; charset=utf-8",
          "mime css");
    check(web_ui_mime_for("_app/version.json") == "application/json", "mime json");
    check(web_ui_mime_for("manifest.webmanifest") == "application/manifest+json", "mime manifest");
    check(web_ui_mime_for("favicon.svg") == "image/svg+xml", "mime svg");
    check(web_ui_mime_for("pwa-512x512.png") == "image/png", "mime png");
    check(web_ui_mime_for("favicon.ico") == "image/x-icon", "mime ico");
    check(web_ui_mime_for("robots.txt") == "text/plain; charset=utf-8", "mime txt");
    check(web_ui_mime_for("fonts/inter.woff2") == "font/woff2", "mime woff2");
    check(web_ui_mime_for("LICENSE") == "application/octet-stream", "mime extensionless");
    check(web_ui_mime_for("archive.tar.zst") == "application/octet-stream", "mime unknown");
}

void check_asset_names() {
    check(web_ui_asset_name("/") == "index.html", "root resolves to index");
    check(web_ui_asset_name("/index.html") == "index.html", "index path resolves to index");
    check(web_ui_asset_name("/sw.js") == "sw.js", "leading slash stripped");
    check(web_ui_asset_name("/_app/immutable/bundle.js") == "_app/immutable/bundle.js",
          "nested path preserved");
    check(web_ui_asset_name("/../etc/passwd").empty(), "dot-dot rejected");
    check(web_ui_asset_name("/_app/../../etc/passwd").empty(), "embedded dot-dot rejected");
    check(web_ui_asset_name("").empty(), "empty path rejected");
    check(web_ui_asset_name("index.html").empty(), "relative request path rejected");
}

void check_table() {
    const WebUiAsset* index = web_ui_find_asset("index.html");
    check(index != nullptr, "index.html is embedded");
    if (index != nullptr) {
        check(index->size > 0, "index.html is non-empty");
        check(index->data != nullptr, "index.html has bytes");
        check(index->content_type == web_ui_mime_for(index->name), "index.html type matches table");
        check(index->etag.size() > 2 && index->etag.front() == '"' && index->etag.back() == '"',
              "index.html etag is a quoted validator");
        const std::string_view body(reinterpret_cast<const char*>(index->data), index->size);
        check(body.find("<!doctype html>") != std::string_view::npos, "index.html is a document");
    }
    check(web_ui_find_asset("index.html") == web_ui_find_asset("index.html"),
          "lookup is stable");
    check(web_ui_find_asset("no-such-asset.bin") == nullptr, "missing asset is not found");
    check(web_ui_find_asset("") == nullptr, "empty name is not found");
    check(web_ui_find_asset("/index.html") == nullptr, "table keys carry no leading slash");

    // Every entry agrees with the shared MIME table and carries a distinct name.
    std::size_t manifests = 0;
    for (const WebUiAsset& asset : web_ui_assets()) {
        check(asset.content_type == web_ui_mime_for(asset.name), "table type matches mime table");
        check(asset.size > 0, "table entry is non-empty");
        check(web_ui_find_asset(asset.name) == &asset, "table entry is reachable by name");
        if (asset.name.ends_with(".webmanifest")) { ++manifests; }
    }
    check(manifests == 1, "the PWA manifest is embedded");
}

void check_cache_control() {
    check(web_ui_cache_control("_app/immutable/bundle.CRRRM4bk.js") ==
              "public, max-age=31536000, immutable",
          "hashed bundle is immutable");
    check(web_ui_cache_control("_app/immutable/assets/bundle.3BIcW1B_.css") ==
              "public, max-age=31536000, immutable",
          "hashed stylesheet is immutable");
    check(web_ui_cache_control("index.html") == "no-cache", "index is revalidated");
    check(web_ui_cache_control("sw.js") == "no-cache", "service worker is revalidated");
    check(web_ui_cache_control("manifest.webmanifest") == "no-cache", "manifest is revalidated");
    check(web_ui_cache_control("_app/version.json") == "no-cache", "version is revalidated");
    check(web_ui_cache_control("build.json") == "no-cache", "build stamp is revalidated");
    check(web_ui_cache_control("pwa-512x512.png") == "public, max-age=3600", "icons are cached");
}

void check_api_paths() {
    for (const char* path : {"/v1/chat/completions", "/v1/models", "/v1/messages", "/v1",
                             "/health", "/metrics", "/slots", "/props"}) {
        check(web_ui_is_api_path(path), "server-owned path");
    }
    for (const char* path : {"/", "/index.html", "/chat/abc", "/sw.js", "/v1beta/chat",
                             "/healthz", "/_app/immutable/bundle.js"}) {
        check(!web_ui_is_api_path(path), "not a server-owned path");
    }
}

void check_spa_fallback() {
    constexpr std::string_view browser =
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";

    check(web_ui_should_serve_index("GET", "/chat/abc", browser), "client-side route falls back");
    check(web_ui_should_serve_index("GET", "/settings", browser), "unknown route falls back");
    check(web_ui_should_serve_index("HEAD", "/chat/abc", browser), "HEAD falls back");

    // API paths never become the SPA document, whatever the client accepts.
    for (const char* path : {"/v1/chat/completions", "/v1/models/nope", "/v1/messages", "/health",
                             "/metrics", "/slots", "/props"}) {
        check(!web_ui_should_serve_index("GET", path, browser), "API path never falls back");
    }

    check(!web_ui_should_serve_index("POST", "/chat/abc", browser), "POST never falls back");
    check(!web_ui_should_serve_index("GET", "/chat/abc", "application/json"),
          "JSON client never falls back");
    check(!web_ui_should_serve_index("GET", "/chat/abc", ""), "no Accept never falls back");
    check(!web_ui_should_serve_index("GET", "/sw.js", browser),
          "an embedded file is served as itself");
    check(!web_ui_should_serve_index("GET", "/index.html", browser),
          "index is served as itself");
    check(web_ui_accepts_html("*/*"), "wildcard accepts html");
    check(!web_ui_accepts_html("application/json, text/plain"), "json client rejects html");

    // Auth exemption: static documents load without the key, the API does not.
    check(web_ui_is_public_request("GET", "/"), "root is public");
    check(web_ui_is_public_request("GET", "/_app/immutable/bundle.js"), "assets are public");
    check(web_ui_is_public_request("GET", "/chat/abc"), "SPA routes are public");
    check(!web_ui_is_public_request("GET", "/props"), "props stays authenticated");
    check(!web_ui_is_public_request("GET", "/v1/models"), "API stays authenticated");
    check(!web_ui_is_public_request("POST", "/"), "only reads are public");
}

void check_props() {
    ServeOptions options;
    options.artifact_path   = "/models/qwen3.ninfer";
    options.max_context     = 65536;
    options.max_concurrency = 4;
    options.default_max_tokens = 2048;
    options.enable_vision   = true;

    ninfer::ModelSamplingDefaults sampling;
    sampling.non_thinking.temperature = 0.7F;
    sampling.non_thinking.top_k       = 20;
    sampling.non_thinking.top_p       = 0.8F;
    sampling.non_thinking.min_p       = 0.05F;

    WebUiPropsInput input;
    input.options           = &options;
    input.model_id          = "qwen3.5";
    input.build_info        = "ninfer (qwen3)";
    input.sampling_defaults = &sampling;

    const nlohmann::json props = nlohmann::json::parse(render_web_ui_props(input));

    check(props["model_path"] == "/models/qwen3.ninfer", "props model_path");
    check(props["model_alias"] == "qwen3.5", "props model_alias");
    check(props["build_info"] == "ninfer (qwen3)", "props build_info");
    check(props["total_slots"] == 4, "props total_slots follows --max-concurrency");
    check(props["n_ctx"] == 65536, "props n_ctx follows --max-context");
    check(props.contains("chat_template"), "props carries chat_template");
    check(!props["chat_template"].get<std::string>().empty(), "props chat_template is non-empty");
    check(props["chat_template"].get<std::string>().find("enable_thinking") != std::string::npos,
          "props advertises the enable_thinking kwarg");
    check(props["modalities"]["vision"] == true, "props vision follows --vision");
    check(props["modalities"]["audio"] == false, "props audio");
    check(props["modalities"]["video"] == false, "props video");
    check(props["default_generation_settings"]["n_ctx"] == 65536, "props generation n_ctx");

    const nlohmann::json& params = props["default_generation_settings"]["params"];
    check(params["temperature"] == 0.7F, "props default temperature");
    check(params["top_k"] == 20, "props default top_k");
    check(params["top_p"] == 0.8F, "props default top_p");
    check(params["min_p"] == 0.05F, "props default min_p");
    check(params.contains("presence_penalty"), "props default presence_penalty");
    check(params.contains("frequency_penalty"), "props default frequency_penalty");
    check(params["max_tokens"] == 2048, "props default max_tokens");

    check(props["endpoint_props"] == true, "props endpoint_props");
    check(props["endpoint_slots"] == true, "props endpoint_slots");
    check(props["endpoint_metrics"] == true, "props endpoint_metrics");
    check(!props.contains("role"), "no role keeps the UI in single-model mode");

    // --chat-template and an explicitly disabled thinking mode are both reflected.
    options.chat_template_path = "/etc/ninfer/template.jinja";
    options.enable_thinking    = false;
    const nlohmann::json overridden = nlohmann::json::parse(render_web_ui_props(input));
    const std::string chat_template = overridden["chat_template"].get<std::string>();
    check(chat_template.find("/etc/ninfer/template.jinja") != std::string::npos,
          "props names the chat template override");
    check(chat_template.find("enable_thinking") == std::string::npos,
          "props hides thinking when it is disabled");
}

} // namespace

int main() {
#ifndef NINFER_EMBED_WEBUI
    std::printf("built without NINFER_EMBED_WEBUI; nothing to serve\n");
    check(!web_ui_has_assets(), "stub build has no assets");
    check(web_ui_find_asset("index.html") == nullptr, "stub build finds no index");
    check(!web_ui_should_serve_index("GET", "/chat/abc", "text/html"), "stub build never falls back");
    check(!web_ui_is_public_request("GET", "/"), "stub build exempts nothing from auth");
    return g_failures != 0 ? 1 : 77;
#else
    check(web_ui_has_assets(), "embedded build has assets");
    check_mime_table();
    check_asset_names();
    check_table();
    check_cache_control();
    check_api_paths();
    check_spa_fallback();
    check_props();
    return g_failures != 0 ? 1 : 0;
#endif
}
