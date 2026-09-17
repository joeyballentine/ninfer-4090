#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// True when the build compiled the structured-output backend. When false every GrammarCompiler
// construction throws and the serving layer keeps rejecting non-text response formats.
[[nodiscard]] bool structured_output_available() noexcept;

// One request's position in the token-level automaton of its output language. The position
// advances token by token and can be forked and rolled back, which is what a speculative round
// needs: the round is previewed on a fork, and the fork is discarded or adopted on commit.
class TokenGrammar {
public:
    TokenGrammar(TokenGrammar&&) noexcept;
    TokenGrammar& operator=(TokenGrammar&&) noexcept;
    TokenGrammar(const TokenGrammar&)            = delete;
    TokenGrammar& operator=(const TokenGrammar&) = delete;
    ~TokenGrammar();

    // Independent copy of this position. Both copies share the compiled grammar.
    [[nodiscard]] TokenGrammar fork() const;
    // Advances by one token. False means the token is outside the language; the position is
    // unchanged and the caller must not publish that token.
    [[nodiscard]] bool accept(TokenId token);
    void rollback(std::uint32_t tokens);
    // Licensed next-token bitset over the whole vocabulary, ceil(vocab/32) little-endian words.
    // The span stays valid until the next call on this object.
    [[nodiscard]] std::span<const std::uint32_t> next_token_bitmask();

private:
    class Impl;
    explicit TokenGrammar(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class GrammarCompiler;
};

// Compiles request output languages against one exact vocabulary. Construction indexes the whole
// vocabulary, so a model keeps a single shared instance and every request only compiles its own
// schema.
class GrammarCompiler {
public:
    // `decoded_vocabulary` holds the raw bytes of every id in ascending order, with an empty entry
    // for an unused id, exactly as Tokenizer::decoded_vocabulary reports them.
    GrammarCompiler(std::vector<std::string> decoded_vocabulary,
                    std::span<const TokenId> model_stop_tokens);
    GrammarCompiler(const GrammarCompiler&)            = delete;
    GrammarCompiler& operator=(const GrammarCompiler&) = delete;
    ~GrammarCompiler();

    // `stop_tokens` is the request's resolved terminal-token set; the grammar accepts them where
    // the language is complete. Throws std::invalid_argument for a schema the compiler rejects.
    [[nodiscard]] TokenGrammar compile(const StructuredOutputOptions& options,
                                       std::span<const TokenId> stop_tokens) const;
    [[nodiscard]] std::uint32_t mask_words() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_5::frontend
