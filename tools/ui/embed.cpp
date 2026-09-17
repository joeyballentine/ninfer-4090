// Build-time tool: turn a directory of static web-UI files into one C++ translation unit that
// defines the ninfer_serve asset table (path -> bytes, ETag, MIME type).
//
// usage: ninfer_webui_embed <asset_dir> <out_cpp>
//
// The bytes are emitted as chunked ordinary string literals rather than brace-initialized byte
// lists: the llama.cpp UI bundle is ~9 MB and is almost entirely printable ASCII, so a literal
// stays close to 1:1 in source size where a `0x..,` list would be six times larger and far
// slower to parse.

#include "serve/web_ui_mime.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Asset {
    std::string name; // serve path relative to the UI root, e.g. "_app/immutable/bundle.X.js"
    fs::path path;
};

bool read_file(const fs::path& path, std::vector<unsigned char>& out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { return false; }
    const auto size = file.tellg();
    if (size < 0) { return false; }
    out.resize(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    if (size > 0) { file.read(reinterpret_cast<char*>(out.data()), size); }
    return static_cast<bool>(file);
}

std::uint64_t fnv1a_64(const std::vector<unsigned char>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash = (hash ^ byte) * 1099511628211ULL;
    }
    return hash;
}

// Appends the payload as a run of adjacent string literals. Every escape is emitted whole and
// octal escapes always use three digits, so a following literal digit can never extend one.
void append_literal(std::string& out, const std::vector<unsigned char>& bytes) {
    constexpr std::size_t kChunkChars = 1024;
    if (bytes.empty()) {
        out += "    \"\"";
        return;
    }
    out += "    \"";
    std::size_t emitted = 0;
    char escape[8];
    for (const unsigned char byte : bytes) {
        if (emitted >= kChunkChars) {
            out += "\"\n    \"";
            emitted = 0;
        }
        switch (byte) {
        case '"': out += "\\\""; emitted += 2; break;
        case '\\': out += "\\\\"; emitted += 2; break;
        case '\n': out += "\\n"; emitted += 2; break;
        case '\r': out += "\\r"; emitted += 2; break;
        case '\t': out += "\\t"; emitted += 2; break;
        // Trigraphs were removed in C++17, but GCC still warns about `??=` and friends; escaping
        // every '?' keeps the generated source warning-free.
        case '?': out += "\\?"; emitted += 2; break;
        default:
            if (byte >= 0x20 && byte < 0x7F) {
                out += static_cast<char>(byte);
                emitted += 1;
            } else {
                std::snprintf(escape, sizeof(escape), "\\%03o", static_cast<unsigned>(byte));
                out += escape;
                emitted += 4;
            }
            break;
        }
    }
    out += '"';
}

// The release tarball wraps everything in a `llama-<tag>/` directory; serve paths are relative to
// whichever directory actually holds index.html.
fs::path find_asset_root(const fs::path& base) {
    std::error_code error;
    if (!fs::exists(base, error)) { return base; }
    if (fs::exists(base / "index.html", error)) { return base; }
    for (const auto& entry : fs::recursive_directory_iterator(base, error)) {
        if (entry.is_regular_file(error) && entry.path().filename() == "index.html") {
            return entry.path().parent_path();
        }
    }
    return base;
}

bool write_if_different(const std::string& path, const std::string& content) {
    std::ifstream existing(path, std::ios::binary);
    if (existing.is_open()) {
        const std::string previous((std::istreambuf_iterator<char>(existing)),
                                   std::istreambuf_iterator<char>());
        if (previous == content) { return true; }
    }
    existing.close();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::fprintf(stderr, "webui embed: cannot write %s\n", path.c_str());
        return false;
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return static_cast<bool>(out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <asset_dir> <out_cpp>\n", argv[0]);
        return 2;
    }
    const fs::path asset_root = find_asset_root(argv[1]);
    const std::string out_cpp = argv[2];

    std::error_code error;
    if (!fs::exists(asset_root / "index.html", error)) {
        std::fprintf(stderr, "webui embed: no index.html under %s\n", asset_root.string().c_str());
        return 1;
    }

    std::vector<Asset> assets;
    for (const auto& entry : fs::recursive_directory_iterator(asset_root, error)) {
        if (!entry.is_regular_file(error)) { continue; }
        std::string name = entry.path().lexically_relative(asset_root).generic_string();
        if (name.empty() || name.starts_with("..") || name.front() == '.') { continue; }
        assets.push_back(Asset{std::move(name), entry.path()});
    }
    if (error) {
        std::fprintf(stderr, "webui embed: cannot walk %s: %s\n", asset_root.string().c_str(),
                     error.message().c_str());
        return 1;
    }
    std::sort(assets.begin(), assets.end(),
              [](const Asset& lhs, const Asset& rhs) { return lhs.name < rhs.name; });

    std::string out;
    out.reserve(16u << 20);
    out += "// Generated by tools/ui/embed.cpp from the pinned llama.cpp web UI release.\n";
    out += "// Do not edit; re-run the build to regenerate.\n\n";
    out += "#include \"serve/web_ui.h\"\n\n";
    out += "namespace ninfer::serve {\n";
    out += "namespace {\n\n";

    std::string table;
    char line[1024];
    for (std::size_t index = 0; index < assets.size(); ++index) {
        std::vector<unsigned char> bytes;
        if (!read_file(assets[index].path, bytes)) {
            std::fprintf(stderr, "webui embed: cannot read %s\n",
                         assets[index].path.string().c_str());
            return 1;
        }
        std::snprintf(line, sizeof(line), "const unsigned char kAssetData%zu[] =\n", index);
        out += line;
        append_literal(out, bytes);
        out += ";\n\n";

        const std::string_view mime = ninfer::serve::web_ui_mime_for(assets[index].name);
        std::snprintf(line, sizeof(line),
                      "    {\"%s\", kAssetData%zu, %zuu, \"\\\"%016" PRIx64 "\\\"\", \"%.*s\"},\n",
                      assets[index].name.c_str(), index, bytes.size(), fnv1a_64(bytes),
                      static_cast<int>(mime.size()), mime.data());
        table += line;
    }

    out += "const WebUiAsset kAssets[] = {\n";
    out += table;
    out += "};\n\n";
    out += "} // namespace\n\n";
    out += "bool web_ui_has_assets() noexcept { return true; }\n\n";
    out += "std::span<const WebUiAsset> web_ui_assets() noexcept {\n";
    out += "    return std::span<const WebUiAsset>(kAssets, std::size(kAssets));\n";
    out += "}\n\n";
    out += "} // namespace ninfer::serve\n";

    if (!write_if_different(out_cpp, out)) { return 1; }
    std::printf("webui embed: %zu assets -> %s\n", assets.size(), out_cpp.c_str());
    return 0;
}
