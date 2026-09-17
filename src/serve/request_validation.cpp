#include "serve/request_validation.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

ResponseFormat parse_response_format_object(const RequestJson& format,
                                            const RequestJson& definition, std::string_view param,
                                            std::string_view definition_param) {
    const std::string path(param);
    const std::string body(definition_param);
    if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
        bad_request(path + " must be an object with a string type", path);
    }
    const std::string type = format.at("type").get<std::string>();
    ResponseFormat out;
    if (type == "text") { return out; }
    if (type != "json_object" && type != "json_schema") {
        bad_request(path + ".type must be text, json_object, or json_schema", path + ".type",
                    "response_format_type_invalid");
    }
#if !NINFER_STRUCTURED_OUTPUT
    bad_request("this response format requires constrained output, which this build does not "
                "provide; only {\"type\":\"text\"} is available",
                path, "response_format_not_supported");
#else
    if (type == "json_object") {
        out.mode = ResponseFormatMode::JsonObject;
        return out;
    }
    if (!definition.contains("name") || !definition.at("name").is_string() ||
        definition.at("name").get_ref<const std::string&>().empty()) {
        bad_request(body + ".name must be a non-empty string", body + ".name");
    }
    if (!definition.contains("schema") || !definition.at("schema").is_object()) {
        bad_request(body + ".schema must be a JSON Schema object", body + ".schema");
    }
    if (definition.contains("strict") && !definition.at("strict").is_null() &&
        !definition.at("strict").is_boolean()) {
        bad_request(body + ".strict must be a boolean", body + ".strict");
    }
    out.mode        = ResponseFormatMode::JsonSchema;
    out.name        = definition.at("name").get<std::string>();
    out.schema_json = definition.at("schema").dump();
    out.strict      = definition.contains("strict") && definition.at("strict").is_boolean() &&
                 definition.at("strict").get<bool>();
    return out;
#endif
}

void reject_structured_output_conflicts(const GenerationRequest& request) {
    if (!request.response_format.constrained() || !request.uses_tools()) { return; }
    bad_request("a structured response format cannot be combined with tool calls", "tools",
                "response_format_conflict");
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

} // namespace ninfer::serve
