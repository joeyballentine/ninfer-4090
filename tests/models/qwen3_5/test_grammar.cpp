// Host-only contract test for the structured-output grammar. It compiles a small schema against a
// synthetic byte vocabulary, then checks the two properties the sampler and the Engine depend on:
// the published bitmask licenses exactly the ids that continue the language, and accepting a token
// outside it fails without moving the position. No Engine, model artifact, or GPU is involved.

#include "models/qwen3_5/frontend/grammar.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
using ninfer::TokenId;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

// Ids are positional: the test refers to them by name below.
enum Token : std::int32_t {
    kBraceOpen = 0,
    kBraceClose,
    kQuote,
    kColon,
    kComma,
    kSpace,
    kOk,
    kTrue,
    kFalse,
    kDigit7,
    kStop,
    kVocabSize,
};

std::vector<std::string> vocabulary() {
    std::vector<std::string> vocab(kVocabSize);
    vocab[kBraceOpen]  = "{";
    vocab[kBraceClose] = "}";
    vocab[kQuote]      = "\"";
    vocab[kColon]      = ":";
    vocab[kComma]      = ",";
    vocab[kSpace]      = " ";
    vocab[kOk]         = "ok";
    vocab[kTrue]       = "true";
    vocab[kFalse]      = "false";
    vocab[kDigit7]     = "7";
    vocab[kStop]       = "<|im_end|>";
    return vocab;
}

bool licensed(std::span<const std::uint32_t> mask, std::int32_t token) {
    const auto id = static_cast<std::uint32_t>(token);
    return (mask[id >> 5] & (1U << (id & 31U))) != 0U;
}

int object_grammar_contract() {
    const std::vector<TokenId> stops{kStop};
    fi::GrammarCompiler compiler(vocabulary(), stops);

    ninfer::StructuredOutputOptions options;
    options.mode = ninfer::StructuredOutputMode::JsonObject;
    fi::TokenGrammar grammar = compiler.compile(options, stops);

    int failures = 0;
    std::span<const std::uint32_t> mask = grammar.next_token_bitmask();
    failures += check(mask.size() == (kVocabSize + 31) / 32, "bitmask covers the whole vocabulary");
    failures += check(licensed(mask, kBraceOpen), "a JSON object may start with '{'");
    failures += check(!licensed(mask, kBraceClose), "a JSON object may not start with '}'");
    failures += check(!licensed(mask, kStop), "an empty response is not valid JSON");

    failures += check(!grammar.accept(kBraceClose), "an unlicensed token is rejected");
    failures += check(licensed(grammar.next_token_bitmask(), kBraceOpen),
                      "a rejected token leaves the position unchanged");

    failures += check(grammar.accept(kBraceOpen), "'{' is accepted");
    failures += check(licensed(grammar.next_token_bitmask(), kBraceClose),
                      "an empty object may close immediately");
    return failures;
}

int schema_grammar_contract() {
    const std::vector<TokenId> stops{kStop};
    fi::GrammarCompiler compiler(vocabulary(), stops);

    ninfer::StructuredOutputOptions options;
    options.mode        = ninfer::StructuredOutputMode::JsonSchema;
    options.name        = "flag";
    options.strict      = true;
    options.schema_json = R"({"type":"object",)"
                          R"("properties":{"ok":{"type":"boolean"}},)"
                          R"("required":["ok"],"additionalProperties":false})";
    fi::TokenGrammar grammar = compiler.compile(options, stops);

    int failures                          = 0;
    const std::int32_t opening[]          = {kBraceOpen, kQuote, kOk, kQuote, kColon};
    for (const std::int32_t token : opening) {
        failures += check(licensed(grammar.next_token_bitmask(), token),
                          "the only schema-licensed prefix continues");
        failures += check(grammar.accept(token), "the schema-licensed prefix is accepted");
    }
    const std::span<const std::uint32_t> after_colon = grammar.next_token_bitmask();
    failures += check(licensed(after_colon, kTrue) && licensed(after_colon, kFalse),
                      "a boolean property licenses both boolean literals");
    failures += check(!licensed(after_colon, kDigit7) && !licensed(after_colon, kQuote),
                      "a boolean property licenses no number or string");

    failures += check(grammar.accept(kTrue) && grammar.accept(kBraceClose),
                      "the complete document is accepted");
    failures += check(licensed(grammar.next_token_bitmask(), kStop),
                      "a completed document licenses the terminal token");

    // A speculative round previews on a fork and rewinds what it did not publish.
    fi::TokenGrammar preview = grammar.fork();
    failures += check(preview.accept(kStop), "the fork accepts the terminal token");
    preview.rollback(1);
    failures += check(licensed(preview.next_token_bitmask(), kStop),
                      "a rolled-back fork returns to the committed position");
    failures += check(licensed(grammar.next_token_bitmask(), kStop),
                      "forking does not move the committed position");
    return failures;
}

int rejected_schema_contract() {
    const std::vector<TokenId> stops{kStop};
    fi::GrammarCompiler compiler(vocabulary(), stops);
    ninfer::StructuredOutputOptions options;
    options.mode        = ninfer::StructuredOutputMode::JsonSchema;
    options.name        = "broken";
    options.schema_json = R"({"type":)";
    bool threw          = false;
    try {
        (void)compiler.compile(options, stops);
    } catch (const std::invalid_argument&) { threw = true; }
    return check(threw, "a malformed schema is reported as an invalid argument");
}

} // namespace

int main() {
    if (!fi::structured_output_available()) {
        std::cout << "SKIP structured output is not compiled in\n";
        return 77;
    }
    int failures = 0;
    failures += object_grammar_contract();
    failures += schema_grammar_contract();
    failures += rejected_schema_contract();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen3_5 structured output grammar\n";
    return failures == 0 ? 0 : 1;
}
