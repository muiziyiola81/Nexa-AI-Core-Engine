/**
 * NEXA Native Runtime - Tokenizer Interface Header
 * Defines vocabulary mapping, token encoding, decoding, special tokens,
 * byte-level tokenizer, and trainable Byte Pair Encoding (BPE) tokenizer.
 */

#ifndef NEXA_TOKENIZER_HPP
#define NEXA_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace nexa {

struct SpecialTokens {
    int32_t bosId = 1; // <|begin_of_text|> or <s>
    int32_t eosId = 2; // <|end_of_text|> or </s>
    int32_t unkId = 0; // <unk>
    int32_t padId = 3; // <pad>
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text, bool addBos = true, bool addEos = false) = 0;
    virtual std::string decode(const std::vector<int32_t>& tokens, bool skipSpecial = true) = 0;
    virtual size_t vocabSize() const = 0;
    virtual const SpecialTokens& specialTokens() const = 0;
};

class NexaByteTokenizer : public Tokenizer {
private:
    std::unordered_map<std::string, int32_t> tokenToId_;
    std::vector<std::string> idToToken_;
    SpecialTokens special_;

public:
    NexaByteTokenizer();

    void addToken(const std::string& token, int32_t id);
    void loadFromVocabulary(const std::vector<std::string>& vocab);

    std::vector<int32_t> encode(const std::string& text, bool addBos = true, bool addEos = false) override;
    std::string decode(const std::vector<int32_t>& tokens, bool skipSpecial = true) override;

    size_t vocabSize() const override { return idToToken_.size(); }
    const SpecialTokens& specialTokens() const override { return special_; }
};

struct BpeMerge {
    int32_t tokenA = 0;
    int32_t tokenB = 0;
    int32_t mergedId = 0;
    int32_t rank = 0;
};

class NexaBpeTokenizer : public Tokenizer {
private:
    SpecialTokens special_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int32_t> tokenToId_;
    std::vector<BpeMerge> merges_;
    std::unordered_map<uint64_t, BpeMerge> mergeTable_;

    static uint64_t makePairKey(int32_t a, int32_t b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(b));
    }

public:
    NexaBpeTokenizer();

    /**
     * Trains BPE merge rules on raw corpus text up to targetVocabSize.
     */
    void train(const std::string& corpusText, uint32_t targetVocabSize = 512, uint32_t minFrequency = 2);

    /**
     * Serializes vocabulary and merge table to a binary .nbt file.
     */
    bool save(const std::string& filePath) const;

    /**
     * Deserializes vocabulary and merge table from a binary .nbt file.
     */
    bool load(const std::string& filePath);

    std::vector<int32_t> encode(const std::string& text, bool addBos = true, bool addEos = false) override;
    std::string decode(const std::vector<int32_t>& tokens, bool skipSpecial = true) override;

    size_t vocabSize() const override { return idToToken_.size(); }
    const SpecialTokens& specialTokens() const override { return special_; }
    size_t numMerges() const { return merges_.size(); }
    const std::vector<BpeMerge>& getMerges() const { return merges_; }
};

} // namespace nexa

#endif // NEXA_TOKENIZER_HPP
