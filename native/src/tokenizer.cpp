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
std::vector<std::string> pretokenizeText(const std::string& text) {
    std::vector<std::string> chunks;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        if (text[i] == '\n' || text[i] == '\r') {
            chunks.emplace_back(1, text[i]);
            i++;
            continue;
        }
        size_t start = i;
        if (text[i] == ' ') {
            i++;
        }
        if (i < n && text[i] != ' ' && text[i] != '\n' && text[i] != '\r') {
            while (i < n && text[i] != ' ' && text[i] != '\n' && text[i] != '\r') {
                i++;
            }
        } else {
            while (i < n && text[i] == ' ') {
                i++;
            }
        }
        chunks.push_back(text.substr(start, i - start));
    }
    return chunks;
}
} // namespace

void NexaBpeTokenizer::train(const std::string& corpusText, uint32_t targetVocabSize, uint32_t minFrequency) {
    if (corpusText.empty() || targetVocabSize <= idToToken_.size()) {
        return;
    }

    // Build frequency map over pre-tokenized chunks for fast O(unique_chunks) BPE training
    auto rawChunks = pretokenizeText(corpusText);
    std::unordered_map<std::string, uint32_t> chunkCounts;
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

    // Iterative BPE pair frequency counting and merging
    while (idToToken_.size() < targetVocabSize) {
        std::unordered_map<uint64_t, uint32_t> pairFreq;
        for (const auto& cs : sequences) {
            if (cs.tokens.size() < 2) continue;
            for (size_t i = 0; i + 1 < cs.tokens.size(); ++i) {
                uint64_t key = makePairKey(cs.tokens[i], cs.tokens[i + 1]);
                pairFreq[key] += cs.count;
            }
        }

        if (pairFreq.empty()) break;

        uint64_t bestPair = 0;
        uint32_t bestCount = 0;
        for (const auto& [pair, count] : pairFreq) {
            if (count > bestCount) {
                bestCount = count;
                bestPair = pair;
            }
        }

        if (bestCount < minFrequency) {
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

        // Apply merge over unique chunk sequences
        for (auto& cs : sequences) {
            if (cs.tokens.size() < 2) continue;
            bool changed = false;
            for (size_t i = 0; i + 1 < cs.tokens.size(); ++i) {
                if (cs.tokens[i] == tokA && cs.tokens[i + 1] == tokB) {
                    changed = true;
                    break;
                }
            }
            if (!changed) continue;

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
