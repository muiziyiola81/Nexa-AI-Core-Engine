/**
 * NEXA Native Runtime - Tokenizer Implementation
 * Byte-level tokenizer and trainable Byte Pair Encoding (BPE) tokenizer.
 */

#include "../include/nexa/tokenizer.hpp"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <climits>

namespace nexa {

// ============================================================================
// NexaByteTokenizer Implementation
// ============================================================================

NexaByteTokenizer::NexaByteTokenizer() {
    // Reserve special tokens
    special_.unkId = 0;
    special_.bosId = 1;
    special_.eosId = 2;
    special_.padId = 3;

    addToken("<unk>", special_.unkId);
    addToken("<|begin_of_text|>", special_.bosId);
    addToken("<|end_of_text|>", special_.eosId);
    addToken("<pad>", special_.padId);

    // Initialize ASCII single-byte tokens (4..259)
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        int32_t id = static_cast<int32_t>(idToToken_.size());
        addToken(s, id);
    }
}

void NexaByteTokenizer::addToken(const std::string& token, int32_t id) {
    if (id >= static_cast<int32_t>(idToToken_.size())) {
        idToToken_.resize(id + 1);
    }
    idToToken_[id] = token;
    tokenToId_[token] = id;
}

void NexaByteTokenizer::loadFromVocabulary(const std::vector<std::string>& vocab) {
    for (size_t i = 0; i < vocab.size(); ++i) {
        addToken(vocab[i], static_cast<int32_t>(i));
    }
}

std::vector<int32_t> NexaByteTokenizer::encode(const std::string& text, bool addBos, bool addEos) {
    std::vector<int32_t> tokens;
    if (addBos) tokens.push_back(special_.bosId);

    // Greedy word / byte matching
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        // Try multi-byte substrings up to 32 chars
        size_t maxLen = std::min(static_cast<size_t>(32), text.size() - i);
        for (size_t len = maxLen; len >= 2; --len) {
            std::string sub = text.substr(i, len);
            auto it = tokenToId_.find(sub);
            if (it != tokenToId_.end()) {
                tokens.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }

        if (!matched) {
            // Byte-level fallback
            std::string singleChar(1, text[i]);
            auto it = tokenToId_.find(singleChar);
            if (it != tokenToId_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(special_.unkId);
            }
            i += 1;
        }
    }

    if (addEos) tokens.push_back(special_.eosId);
    return tokens;
}

std::string NexaByteTokenizer::decode(const std::vector<int32_t>& tokens, bool skipSpecial) {
    std::ostringstream oss;
    for (int32_t id : tokens) {
        if (skipSpecial) {
            if (id == special_.bosId || id == special_.eosId || id == special_.padId || id == special_.unkId) {
                continue;
            }
        }
        if (id >= 0 && id < static_cast<int32_t>(idToToken_.size())) {
            oss << idToToken_[id];
        }
    }
    return oss.str();
}

// ============================================================================
// NexaBpeTokenizer Implementation
// ============================================================================

NexaBpeTokenizer::NexaBpeTokenizer() {
    special_.unkId = 0;
    special_.bosId = 1;
    special_.eosId = 2;
    special_.padId = 3;

    // 0..3: Special tokens
    idToToken_.push_back("<unk>");
    idToToken_.push_back("<|begin_of_text|>");
    idToToken_.push_back("<|end_of_text|>");
    idToToken_.push_back("<pad>");

    tokenToId_["<unk>"] = 0;
    tokenToId_["<|begin_of_text|>"] = 1;
    tokenToId_["<|end_of_text|>"] = 2;
    tokenToId_["<pad>"] = 3;

    // 4..259: Initial 256 single byte tokens
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(static_cast<unsigned char>(i)));
        int32_t id = static_cast<int32_t>(idToToken_.size());
        idToToken_.push_back(s);
        tokenToId_[s] = id;
    }
}

namespace {

bool isContractionAt(const std::string& text, size_t i, size_t& outLen) {
    const size_t n = text.size();
    if (i >= n || text[i] != '\'') return false;
    if (i + 2 < n) {
        char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[i + 1])));
        char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[i + 2])));
        bool nextAlpha = (i + 3 < n) && std::isalpha(static_cast<unsigned char>(text[i + 3]));
        if (!nextAlpha && ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))) {
            outLen = 3;
            return true;
        }
    }
    if (i + 1 < n) {
        char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[i + 1])));
        bool nextAlpha = (i + 2 < n) && std::isalpha(static_cast<unsigned char>(text[i + 2]));
        if (!nextAlpha && (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')) {
            outLen = 2;
            return true;
        }
    }
    return false;
}

std::vector<std::string> pretokenizeText(const std::string& text) {
    std::vector<std::string> chunks;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // 1. Line breaks always stay isolated so turn boundaries encode consistently
        if (c == '\n' || c == '\r') {
            chunks.emplace_back(1, text[i]);
            i++;
            continue;
        }

        // 2. English contractions ('s, 't, 're, 've, 'm, 'll, 'd)
        size_t contrLen = 0;
        if (isContractionAt(text, i, contrLen)) {
            chunks.push_back(text.substr(i, contrLen));
            i += contrLen;
            continue;
        }

        // 3. Optional single leading space attached ONLY to ASCII letter words
        if (c == ' ' && i + 1 < n && std::isalpha(static_cast<unsigned char>(text[i + 1]))) {
            size_t start = i;
            i += 2;
            while (i < n && std::isalpha(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // 4. ASCII letter words without leading space
        if (std::isalpha(c)) {
            size_t start = i;
            i++;
            while (i < n && std::isalpha(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // 5. Numbers split into short <= 3-digit runs for clean arithmetic tokenization
        if (std::isdigit(c)) {
            size_t start = i;
            size_t count = 0;
            while (i < n && count < 3 && std::isdigit(static_cast<unsigned char>(text[i]))) {
                i++;
                count++;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // 6. Contiguous horizontal whitespace runs (spaces/tabs)
        if (c == ' ' || c == '\t') {
            size_t start = i;
            while (i < n && (text[i] == ' ' || text[i] == '\t')) {
                // Leave a single space attached if it precedes an alphabetic word
                if (text[i] == ' ' && i > start && i + 1 < n &&
                    std::isalpha(static_cast<unsigned char>(text[i + 1]))) {
                    break;
                }
                i++;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // 7. Non-ASCII UTF-8 sequence (consume full multi-byte codepoint if valid lead byte)
        if ((c & 0x80) != 0) {
            size_t start = i;
            i++;
            while (i < n && (static_cast<unsigned char>(text[i]) & 0xC0) == 0x80) {
                i++;
            }
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // 8. Individual ASCII punctuation/symbol character
        chunks.emplace_back(1, text[i]);
        i++;
    }
    return chunks;
}

} // namespace

void NexaBpeTokenizer::train(const std::string& corpusText, uint32_t targetVocabSize, uint32_t minFrequency) {
    if (corpusText.empty() || targetVocabSize <= idToToken_.size()) {
        return;
    }

    // Build frequency map over pre-tokenized chunks for fast BPE training
    auto rawChunks = pretokenizeText(corpusText);
    std::unordered_map<std::string, uint32_t> chunkCounts;
    chunkCounts.reserve(rawChunks.size() / 4 + 16);
    for (const auto& c : rawChunks) {
        chunkCounts[c]++;
    }

    struct ChunkSeq {
        std::vector<int32_t> tokens;
        uint32_t count;
    };
    std::vector<ChunkSeq> sequences;
    sequences.reserve(chunkCounts.size());
    for (const auto& [str, cnt] : chunkCounts) {
        ChunkSeq cs;
        cs.count = cnt;
        cs.tokens.reserve(str.size());
        for (unsigned char ch : str) {
            cs.tokens.push_back(4 + static_cast<int32_t>(ch));
        }
        sequences.push_back(std::move(cs));
    }

    // Maintain incremental pair frequencies and inverted sequence index
    std::unordered_map<uint64_t, int64_t> pairFreq;
    std::unordered_map<uint64_t, std::vector<size_t>> pairToSeqs;
    pairFreq.reserve(sequences.size() * 2 + 64);
    pairToSeqs.reserve(sequences.size() * 2 + 64);

    for (size_t sIdx = 0; sIdx < sequences.size(); ++sIdx) {
        const auto& toks = sequences[sIdx].tokens;
        const int64_t cnt = static_cast<int64_t>(sequences[sIdx].count);
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            uint64_t key = makePairKey(toks[i], toks[i + 1]);
            pairFreq[key] += cnt;
            auto& vec = pairToSeqs[key];
            if (vec.empty() || vec.back() != sIdx) {
                vec.push_back(sIdx);
            }
        }
    }

    // Iterative BPE pair selection and incremental merge updates
    while (idToToken_.size() < targetVocabSize) {
        uint64_t bestPair = 0;
        int64_t bestCount = 0;
        for (const auto& [pair, count] : pairFreq) {
            if (count > bestCount || (count == bestCount && count > 0 && pair < bestPair)) {
                bestCount = count;
                bestPair = pair;
            }
        }

        if (bestCount < static_cast<int64_t>(minFrequency)) {
            break;
        }

        int32_t tokA = static_cast<int32_t>(bestPair >> 32);
        int32_t tokB = static_cast<int32_t>(bestPair & 0xFFFFFFFF);
        int32_t newId = static_cast<int32_t>(idToToken_.size());

        std::string mergedStr = idToToken_[tokA] + idToToken_[tokB];
        idToToken_.push_back(mergedStr);
        tokenToId_[mergedStr] = newId;

        BpeMerge mergeRule{tokA, tokB, newId, static_cast<int32_t>(merges_.size())};
        merges_.push_back(mergeRule);
        mergeTable_[bestPair] = mergeRule;

        std::vector<size_t> candidateSeqs;
        auto ptIt = pairToSeqs.find(bestPair);
        if (ptIt != pairToSeqs.end()) {
            candidateSeqs = std::move(ptIt->second);
            pairToSeqs.erase(ptIt);
        }
        pairFreq.erase(bestPair);

        // Deduplicate candidate indices once
        std::sort(candidateSeqs.begin(), candidateSeqs.end());
        candidateSeqs.erase(std::unique(candidateSeqs.begin(), candidateSeqs.end()), candidateSeqs.end());

        for (size_t sIdx : candidateSeqs) {
            auto& cs = sequences[sIdx];
            if (cs.tokens.size() < 2) continue;

            bool changed = false;
            for (size_t i = 0; i + 1 < cs.tokens.size(); ++i) {
                if (cs.tokens[i] == tokA && cs.tokens[i + 1] == tokB) {
                    changed = true;
                    break;
                }
            }
            if (!changed) continue;

            const int64_t cnt = static_cast<int64_t>(cs.count);

            // Subtract all previous pairs in this sequence
            for (size_t i = 0; i + 1 < cs.tokens.size(); ++i) {
                uint64_t oldKey = makePairKey(cs.tokens[i], cs.tokens[i + 1]);
                if (oldKey == bestPair) continue;
                auto fIt = pairFreq.find(oldKey);
                if (fIt != pairFreq.end()) {
                    fIt->second -= cnt;
                    if (fIt->second <= 0) {
                        pairFreq.erase(fIt);
                    }
                }
            }

            // Build merged sequence
            std::vector<int32_t> nextTokens;
            nextTokens.reserve(cs.tokens.size());
            for (size_t i = 0; i < cs.tokens.size(); ++i) {
                if (i + 1 < cs.tokens.size() && cs.tokens[i] == tokA && cs.tokens[i + 1] == tokB) {
                    nextTokens.push_back(newId);
                    i++;
                } else {
                    nextTokens.push_back(cs.tokens[i]);
                }
            }
            cs.tokens = std::move(nextTokens);

            // Add all new pairs in merged sequence
            for (size_t i = 0; i + 1 < cs.tokens.size(); ++i) {
                uint64_t newKey = makePairKey(cs.tokens[i], cs.tokens[i + 1]);
                pairFreq[newKey] += cnt;
                auto& vec = pairToSeqs[newKey];
                if (vec.empty() || vec.back() != sIdx) {
                    vec.push_back(sIdx);
                }
            }
        }
    }
}

bool NexaBpeTokenizer::save(const std::string& filePath) const {
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) return false;

    // Header: Magic 0x4E425045 ('NBPE'), version 1
    uint32_t magic = 0x4E425045;
    uint32_t version = 1;
    uint32_t vSize = static_cast<uint32_t>(idToToken_.size());
    uint32_t nMerges = static_cast<uint32_t>(merges_.size());

    out.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&vSize), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&nMerges), sizeof(uint32_t));

    // Write merge table
    for (const auto& m : merges_) {
        out.write(reinterpret_cast<const char*>(&m.tokenA), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&m.tokenB), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&m.mergedId), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&m.rank), sizeof(int32_t));
    }

    // Write vocabulary strings
    for (const auto& s : idToToken_) {
        uint32_t len = static_cast<uint32_t>(s.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
        out.write(s.data(), len);
    }

    return true;
}

bool NexaBpeTokenizer::load(const std::string& filePath) {
    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open()) return false;

    uint32_t magic = 0, version = 0, vSize = 0, nMerges = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&vSize), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&nMerges), sizeof(uint32_t));

    if (magic != 0x4E425045 || version != 1) {
        return false;
    }

    merges_.clear();
    mergeTable_.clear();
    idToToken_.clear();
    tokenToId_.clear();

    merges_.resize(nMerges);
    for (uint32_t i = 0; i < nMerges; ++i) {
        in.read(reinterpret_cast<char*>(&merges_[i].tokenA), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&merges_[i].tokenB), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&merges_[i].mergedId), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&merges_[i].rank), sizeof(int32_t));

        uint64_t key = makePairKey(merges_[i].tokenA, merges_[i].tokenB);
        mergeTable_[key] = merges_[i];
    }

    idToToken_.resize(vSize);
    for (uint32_t i = 0; i < vSize; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
        std::string s(len, '\0');
        in.read(&s[0], len);
        idToToken_[i] = s;
        tokenToId_[s] = static_cast<int32_t>(i);
    }

    return in.good();
}

std::vector<int32_t> NexaBpeTokenizer::encode(const std::string& text, bool addBos, bool addEos) {
    std::vector<int32_t> tokens;
    if (addBos) tokens.push_back(special_.bosId);

    if (text.empty()) {
        if (addEos) tokens.push_back(special_.eosId);
        return tokens;
    }

    auto chunks = pretokenizeText(text);
    tokens.reserve(text.size() / 2 + 4);

    for (const auto& chunk : chunks) {
        std::vector<int32_t> wordTokens;
        wordTokens.reserve(chunk.size());
        for (unsigned char c : chunk) {
            wordTokens.push_back(4 + static_cast<int32_t>(c));
        }

        while (wordTokens.size() >= 2) {
            int32_t bestRank = INT32_MAX;
            size_t bestIdx = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < wordTokens.size(); ++i) {
                uint64_t key = makePairKey(wordTokens[i], wordTokens[i + 1]);
                auto it = mergeTable_.find(key);
                if (it != mergeTable_.end() && it->second.rank < bestRank) {
                    bestRank = it->second.rank;
                    bestIdx = i;
                    found = true;
                }
            }

            if (!found) break;

            uint64_t key = makePairKey(wordTokens[bestIdx], wordTokens[bestIdx + 1]);
            int32_t mergedId = mergeTable_[key].mergedId;
            wordTokens[bestIdx] = mergedId;
            wordTokens.erase(wordTokens.begin() + bestIdx + 1);
        }

        for (int32_t t : wordTokens) {
            tokens.push_back(t);
        }
    }

    if (addEos) tokens.push_back(special_.eosId);
    return tokens;
}

std::string NexaBpeTokenizer::decode(const std::vector<int32_t>& tokens, bool skipSpecial) {
    std::ostringstream oss;
    for (int32_t id : tokens) {
        if (skipSpecial) {
            if (id == special_.bosId || id == special_.eosId || id == special_.padId || id == special_.unkId) {
                continue;
            }
        }
        if (id >= 0 && id < static_cast<int32_t>(idToToken_.size())) {
            oss << idToToken_[id];
        }
    }
    return oss.str();
}

} // namespace nexa
