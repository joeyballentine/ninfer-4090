#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// Reads a `{type, ...}` output-format object into the wire-independent contract. `definition`
// carries the json_schema body (`name`/`schema`/`strict`): Chat Completions nests it under
// `response_format.json_schema` while the Responses API inlines it into `text.format`, so the
// caller supplies the object and the `param` path used in errors. When the build has no
// structured-output backend every non-text type is rejected with response_format_not_supported.
[[nodiscard]] ResponseFormat parse_response_format_object(const RequestJson& format,
                                                          const RequestJson& definition,
                                                          std::string_view param,
                                                          std::string_view definition_param);

// A structured response is its grammar's language from the first generated token, so it has no
// thinking phase to cap and no tool-call grammar to interleave. Reject those combinations at the
// protocol boundary instead of resolving them silently.
void reject_structured_output_conflicts(const GenerationRequest& request);

} // namespace ninfer::serve
