#include "models/qwen3_5/frontend/grammar.h"

#include <stdexcept>
#include <utility>

#if NINFER_STRUCTURED_OUTPUT
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>
#endif

namespace ninfer::models::qwen3_5::frontend {

#if NINFER_STRUCTURED_OUTPUT

bool structured_output_available() noexcept { return true; }

class TokenGrammar::Impl {
public:
    Impl(xgrammar::GrammarMatcher matcher_, std::int32_t mask_words_)
        : matcher(std::move(matcher_)), mask(static_cast<std::size_t>(mask_words_), 0) {}

    xgrammar::GrammarMatcher matcher;
    // xgrammar fills a signed 32-bit bitmask; the Op reads the same words as an unsigned bitset.
    std::vector<std::int32_t> mask;
};

class GrammarCompiler::Impl {
public:
    explicit Impl(xgrammar::TokenizerInfo info, std::int32_t mask_words_)
        : compiler(info), mask_words(mask_words_) {}

    mutable xgrammar::GrammarCompiler compiler;
    std::int32_t mask_words = 0;
};

namespace {

std::vector<int> to_int_vector(std::span<const TokenId> tokens) {
    std::vector<int> out;
    out.reserve(tokens.size());
    for (const TokenId token : tokens) { out.push_back(static_cast<int>(token)); }
    return out;
}

} // namespace

GrammarCompiler::GrammarCompiler(std::vector<std::string> decoded_vocabulary,
                                 std::span<const TokenId> model_stop_tokens) {
    const auto vocab_size = static_cast<int>(decoded_vocabulary.size());
    if (vocab_size <= 0) {
        throw std::invalid_argument("structured output requires a non-empty tokenizer vocabulary");
    }
    // The vocabulary is already decoded to raw bytes, so no further byte-decoding convention
    // applies to it.
    xgrammar::TokenizerInfo info(decoded_vocabulary, xgrammar::VocabType::RAW, vocab_size,
                                 to_int_vector(model_stop_tokens));
    impl_ = std::make_unique<Impl>(std::move(info), xgrammar::GetBitmaskSize(vocab_size));
}

GrammarCompiler::~GrammarCompiler() = default;

std::uint32_t GrammarCompiler::mask_words() const noexcept {
    return static_cast<std::uint32_t>(impl_->mask_words);
}

TokenGrammar GrammarCompiler::compile(const StructuredOutputOptions& options,
                                      std::span<const TokenId> stop_tokens) const {
    if (!options.enabled()) {
        throw std::invalid_argument("structured output compilation requires an output language");
    }
    try {
        xgrammar::CompiledGrammar grammar =
            options.mode == StructuredOutputMode::JsonObject
                ? impl_->compiler.CompileBuiltinJSONGrammar()
                : impl_->compiler.CompileJSONSchema(options.schema_json, true, std::nullopt,
                                                    std::nullopt, options.strict);
        // A negative rollback bound keeps every accepted token rewindable, which a speculative
        // round needs when its preview is truncated or discarded.
        xgrammar::GrammarMatcher matcher(grammar, to_int_vector(stop_tokens), false, -1);
        return TokenGrammar(std::make_unique<TokenGrammar::Impl>(std::move(matcher),
                                                                 impl_->mask_words));
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("output schema cannot be compiled: ") +
                                    error.what());
    }
}

TokenGrammar::TokenGrammar(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

TokenGrammar::TokenGrammar(TokenGrammar&&) noexcept            = default;
TokenGrammar& TokenGrammar::operator=(TokenGrammar&&) noexcept = default;
TokenGrammar::~TokenGrammar()                                  = default;

TokenGrammar TokenGrammar::fork() const {
    return TokenGrammar(std::make_unique<Impl>(impl_->matcher.Fork(),
                                               static_cast<std::int32_t>(impl_->mask.size())));
}

bool TokenGrammar::accept(TokenId token) {
    return impl_->matcher.AcceptToken(static_cast<std::int32_t>(token));
}

void TokenGrammar::rollback(std::uint32_t tokens) {
    if (tokens == 0) { return; }
    impl_->matcher.Rollback(static_cast<int>(tokens));
}

std::span<const std::uint32_t> TokenGrammar::next_token_bitmask() {
    auto shape = static_cast<std::int64_t>(impl_->mask.size());
    DLTensor tensor{};
    tensor.data        = impl_->mask.data();
    tensor.device      = DLDevice{kDLCPU, 0};
    tensor.ndim        = 1;
    tensor.dtype       = xgrammar::GetBitmaskDLType();
    tensor.shape       = &shape;
    tensor.strides     = nullptr;
    tensor.byte_offset = 0;
    (void)impl_->matcher.FillNextTokenBitmask(&tensor);
    return {reinterpret_cast<const std::uint32_t*>(impl_->mask.data()), impl_->mask.size()};
}

#else

bool structured_output_available() noexcept { return false; }

class TokenGrammar::Impl {};

class GrammarCompiler::Impl {};

GrammarCompiler::GrammarCompiler(std::vector<std::string>, std::span<const TokenId>) {
    throw std::invalid_argument("this build does not compile structured output");
}

GrammarCompiler::~GrammarCompiler() = default;

std::uint32_t GrammarCompiler::mask_words() const noexcept { return 0; }

TokenGrammar GrammarCompiler::compile(const StructuredOutputOptions&,
                                      std::span<const TokenId>) const {
    throw std::invalid_argument("this build does not compile structured output");
}

TokenGrammar::TokenGrammar(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

TokenGrammar::TokenGrammar(TokenGrammar&&) noexcept            = default;
TokenGrammar& TokenGrammar::operator=(TokenGrammar&&) noexcept = default;
TokenGrammar::~TokenGrammar()                                  = default;

TokenGrammar TokenGrammar::fork() const { return TokenGrammar(nullptr); }

bool TokenGrammar::accept(TokenId) { return false; }

void TokenGrammar::rollback(std::uint32_t) {}

std::span<const std::uint32_t> TokenGrammar::next_token_bitmask() { return {}; }

#endif

} // namespace ninfer::models::qwen3_5::frontend
