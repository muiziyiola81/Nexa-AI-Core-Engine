/**
 * NEXA Native Training Engine - Dataset Pipeline Implementation
 */

#include "../include/nexa/dataset.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <filesystem>
#include <random>

namespace nexa {

bool DatasetPipeline::validateUtf8(const std::string& data, size_t* outErrorOffset) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data.data());
    size_t len = data.size();
    size_t i = 0;

    while (i < len) {
        uint8_t c = bytes[i];

        if (c <= 0x7F) {
            // 1-byte ASCII (0x00..0x7F)
            i += 1;
        } else if (c >= 0xC2 && c <= 0xDF) {
            // 2-byte sequence
            if (i + 1 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            if ((c2 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 2;
        } else if (c == 0xE0) {
            // 3-byte sequence (E0 A0..BF 80..BF)
            if (i + 2 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            uint8_t c3 = bytes[i + 2];
            if (c2 < 0xA0 || c2 > 0xBF || (c3 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 3;
        } else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
            // Standard 3-byte sequence
            if (i + 2 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            uint8_t c3 = bytes[i + 2];
            if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 3;
        } else if (c == 0xED) {
            // 3-byte sequence excluding UTF-16 surrogates (ED 80..9F 80..BF)
            if (i + 2 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            uint8_t c3 = bytes[i + 2];
            if (c2 < 0x80 || c2 > 0x9F || (c3 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 3;
        } else if (c == 0xF0) {
            // 4-byte sequence (F0 90..BF 80..BF 80..BF)
            if (i + 3 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            uint8_t c3 = bytes[i + 2];
            uint8_t c4 = bytes[i + 3];
            if (c2 < 0x90 || c2 > 0xBF || (c3 & 0xC0) != 0x80 || (c4 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 4;
        } else if (c >= 0xF1 && c <= 0xF3) {
            // Standard 4-byte sequence
            if (i + 3 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            uint8_t c3 = bytes[i + 2];
            uint8_t c4 = bytes[i + 3];
            if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80 || (c4 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 4;
        } else if (c == 0xF4) {
            // 4-byte sequence up to 0x10FFFF (F4 80..8F 80..BF 80..BF)
            if (i + 3 >= len) {
                if (outErrorOffset) *outErrorOffset = i;
                return false;
            }
            uint8_t c2 = bytes[i + 1];
            uint8_t c3 = bytes[i + 2];
            uint8_t c4 = bytes[i + 3];
            if (c2 < 0x80 || c2 > 0x8F || (c3 & 0xC0) != 0x80 || (c4 & 0xC0) != 0x80) {
                if (outErrorOffset) *outErrorOffset = i + 1;
                return false;
            }
            i += 4;
        } else {
            // Invalid leading byte (0x80..0xC1 or 0xF5..0xFF)
            if (outErrorOffset) *outErrorOffset = i;
            return false;
        }
    }

    return true;
}

std::string DatasetPipeline::computeContentHashHex(const std::string& data) {
    uint64_t hash = 14695981039346656037ULL; // FNV-1a 64-bit offset basis
    for (unsigned char c : data) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    static const char* hex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = hex[hash & 0xF];
        hash >>= 4;
    }
    return out;
}

bool DatasetPipeline::isVerifiedPermissiveLicense(const std::string& license) {
    std::string trimmed = license;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) trimmed.erase(trimmed.begin());
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) trimmed.pop_back();
    if (trimmed.empty()) return false;

    std::string lower;
    lower.reserve(trimmed.size());
    for (unsigned char c : trimmed) lower.push_back(static_cast<char>(std::tolower(c)));

    return (lower == "mit" ||
            lower == "apache-2.0" ||
            lower == "bsd-2-clause" ||
            lower == "bsd-3-clause" ||
            lower == "cc-by-4.0" ||
            lower == "cc0-1.0" ||
            lower == "isc");
}

namespace {

std::string trimCopy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(start, end - start);
}

std::string toLowerCopy(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

std::string normalizeAlphaNumSpace(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prevSpace = true;
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
            prevSpace = false;
        } else if (!prevSpace) {
            out.push_back(' ');
            prevSpace = true;
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<std::string> splitWordsNormalized(const std::string& s) {
    std::vector<std::string> words;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            words.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

uint64_t hashString64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<uint64_t> buildWordNgramHashes(const std::vector<std::string>& words, size_t n) {
    std::vector<uint64_t> hashes;
    if (words.empty()) return hashes;
    size_t k = std::min(n, words.size());
    for (size_t i = 0; i + k <= words.size(); ++i) {
        std::string gram;
        for (size_t j = 0; j < k; ++j) {
            if (j > 0) gram.push_back('\x1f');
            gram += words[i + j];
        }
        hashes.push_back(hashString64(gram));
    }
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    return hashes;
}

float computeSortedHashJaccard(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    size_t ia = 0, ib = 0, inter = 0;
    while (ia < a.size() && ib < b.size()) {
        if (a[ia] == b[ib]) {
            inter++;
            ia++;
            ib++;
        } else if (a[ia] < b[ib]) {
            ia++;
        } else {
            ib++;
        }
    }
    size_t uni = a.size() + b.size() - inter;
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

bool containsReplacementOrInvalidCtrl(const std::string& text) {
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x20 && c != '\n' && c != '\r' && c != '\t') {
            return true;
        }
        if (i + 2 < text.size() &&
            static_cast<unsigned char>(text[i]) == 0xEF &&
            static_cast<unsigned char>(text[i + 1]) == 0xBF &&
            static_cast<unsigned char>(text[i + 2]) == 0xBD) {
            return true; // U+FFFD replacement character
        }
    }
    return false;
}

bool parseJsonStringAt(const std::string& s, size_t& pos, std::string& out) {
    if (pos >= s.size() || s[pos] != '"') return false;
    pos++;
    out.clear();
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"') return true;
        if (c == '\\') {
            if (pos >= s.size()) return false;
            char esc = s[pos++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u':
                    if (pos + 4 <= s.size()) {
                        unsigned int cp = 0;
                        std::istringstream hexStream(s.substr(pos, 4));
                        if (hexStream >> std::hex >> cp) {
                            if (cp < 0x80) out.push_back(static_cast<char>(cp));
                            else if (cp < 0x800) {
                                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            } else {
                                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            }
                        }
                        pos += 4;
                    }
                    break;
                default: out.push_back(esc); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

bool extractJsonTopLevelStringField(const std::string& json, const std::string& key, std::string& outVal) {
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t afterKey = pos + needle.size();
        while (afterKey < json.size() && std::isspace(static_cast<unsigned char>(json[afterKey]))) afterKey++;
        if (afterKey < json.size() && json[afterKey] == ':') {
            size_t valStart = afterKey + 1;
            while (valStart < json.size() && std::isspace(static_cast<unsigned char>(json[valStart]))) valStart++;
            if (valStart < json.size() && json[valStart] == '"') {
                size_t cursor = valStart;
                if (parseJsonStringAt(json, cursor, outVal)) {
                    return true;
                }
            }
        }
        pos = afterKey;
    }
    return false;
}

bool hasYesNoPolarity(const std::vector<std::string>& words, int& outPolarity) {
    if (words.empty()) return false;
    if (words[0] == "yes" || words[0] == "true") {
        outPolarity = 1;
        return true;
    }
    if (words[0] == "no" || words[0] == "false") {
        outPolarity = -1;
        return true;
    }
    return false;
}

std::vector<std::string> extractNumbers(const std::string& s) {
    std::vector<std::string> nums;
    std::string cur;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isdigit(c) || (c == '.' && !cur.empty() && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1])))) {
            cur.push_back(static_cast<char>(c));
        } else if (!cur.empty()) {
            nums.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) nums.push_back(cur);
    return nums;
}

} // namespace

bool DatasetPipeline::parseJsonlInstructionLine(
    const std::string& jsonLine,
    std::string& outBlockText,
    std::string& outSource,
    std::string& outLicense,
    std::string* outError
) {
    outBlockText.clear();
    outSource = "unspecified";
    outLicense = "UNVERIFIED";

    std::string trimmed = trimCopy(jsonLine);
    if (trimmed.empty() || trimmed.front() != '{' || trimmed.back() != '}') {
        if (outError) *outError = "JSONL line is not a JSON object";
        return false;
    }

    std::string src, lic;
    if (extractJsonTopLevelStringField(trimmed, "source", src) && !trimCopy(src).empty()) {
        outSource = trimCopy(src);
    }
    if (extractJsonTopLevelStringField(trimmed, "license", lic) && !trimCopy(lic).empty()) {
        outLicense = trimCopy(lic);
    }

    // Mode 1: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    size_t msgPos = trimmed.find("\"messages\"");
    if (msgPos != std::string::npos) {
        size_t arrStart = trimmed.find('[', msgPos);
        if (arrStart == std::string::npos) {
            if (outError) *outError = "JSONL messages field missing '['";
            return false;
        }
        std::vector<std::pair<std::string, std::string>> turns;
        size_t i = arrStart + 1;
        while (i < trimmed.size()) {
            while (i < trimmed.size() && std::isspace(static_cast<unsigned char>(trimmed[i]))) i++;
            if (i >= trimmed.size() || trimmed[i] == ']') break;
            if (trimmed[i] == ',') { i++; continue; }
            if (trimmed[i] != '{') {
                if (outError) *outError = "Malformed message object in JSONL";
                return false;
            }
            size_t objStart = i;
            bool inStr = false;
            bool escaped = false;
            int depth = 0;
            size_t objEnd = std::string::npos;
            for (; i < trimmed.size(); ++i) {
                char ch = trimmed[i];
                if (escaped) { escaped = false; continue; }
                if (ch == '\\' && inStr) { escaped = true; continue; }
                if (ch == '"') { inStr = !inStr; continue; }
                if (!inStr) {
                    if (ch == '{') depth++;
                    else if (ch == '}') {
                        depth--;
                        if (depth == 0) { objEnd = i; i++; break; }
                    }
                }
            }
            if (objEnd == std::string::npos) {
                if (outError) *outError = "Unterminated message object in JSONL";
                return false;
            }
            std::string objSlice = trimmed.substr(objStart, objEnd - objStart + 1);
            std::string role, content;
            if (!extractJsonTopLevelStringField(objSlice, "role", role) ||
                !extractJsonTopLevelStringField(objSlice, "content", content)) {
                if (outError) *outError = "Message missing role or content field";
                return false;
            }
            turns.emplace_back(toLowerCopy(trimCopy(role)), trimCopy(content));
        }
        if (turns.empty()) {
            if (outError) *outError = "JSONL messages array is empty";
            return false;
        }
        std::ostringstream oss;
        for (size_t t = 0; t < turns.size(); ++t) {
            const std::string& role = turns[t].first;
            std::string cleanContent = turns[t].second;
            for (char& ch : cleanContent) {
                if (ch == '\n' || ch == '\r') ch = ' ';
            }
            if (role == "user") {
                oss << "User: " << cleanContent;
            } else if (role == "assistant") {
                oss << "Assistant: " << cleanContent;
            } else {
                if (outError) *outError = "Unsupported message role: " + role;
                return false;
            }
            if (t + 1 < turns.size()) oss << "\n";
        }
        outBlockText = oss.str();
        return true;
    }

    // Mode 2: {"prompt": "...", "response": "..."} or {"question": "...", "answer": "..."}
    std::string prompt, response;
    bool hasP = extractJsonTopLevelStringField(trimmed, "prompt", prompt) ||
                extractJsonTopLevelStringField(trimmed, "question", prompt);
    bool hasR = extractJsonTopLevelStringField(trimmed, "response", response) ||
                extractJsonTopLevelStringField(trimmed, "answer", response);
    if (hasP && hasR) {
        prompt = trimCopy(prompt);
        response = trimCopy(response);
        for (char& ch : prompt) if (ch == '\n' || ch == '\r') ch = ' ';
        for (char& ch : response) if (ch == '\n' || ch == '\r') ch = ' ';
        outBlockText = "User: " + prompt + "\nAssistant: " + response;
        return true;
    }

    if (outError) *outError = "JSONL line missing messages or prompt/response fields";
    return false;
}

DatasetQualityReport DatasetPipeline::analyzeAndFilterBlocks(
    const std::vector<RawExampleBlock>& inputBlocks,
    std::vector<RawExampleBlock>& outAcceptedBlocks,
    float nearDuplicateThreshold,
    bool filterFailed
) {
    outAcceptedBlocks.clear();
    DatasetQualityReport report;
    report.totalInputBlocks = static_cast<uint32_t>(inputBlocks.size());

    std::unordered_set<std::string> exactNormSeen;
    std::unordered_map<uint64_t, std::vector<size_t>> ngramToAccepted;
    std::unordered_map<uint64_t, std::vector<size_t>> qNgramToAccepted;

    struct AcceptedRecord {
        std::vector<uint64_t> blockNgrams;
        std::vector<uint64_t> qNgrams;
        std::string firstQuestion;
        std::string firstAnswer;
    };
    std::vector<AcceptedRecord> acceptedMeta;

    auto addIssue = [&](size_t idx, const std::string& srcFile, const std::string& type, const std::string& detail) {
        if (report.issues.size() < 200) {
            report.issues.push_back({idx, srcFile, type, detail});
        }
    };

    for (size_t idx = 0; idx < inputBlocks.size(); ++idx) {
        const auto& blk = inputBlocks[idx];
        std::string blockText = trimCopy(blk.text);
        if (blockText.empty()) {
            report.rejectedBlocks++;
            report.lowInformationCount++;
            addIssue(idx, blk.sourceFile, "low_information", "Empty block");
            continue;
        }

        // 1. UTF-8 and Corrupted Text Check
        size_t errOff = 0;
        if (!validateUtf8(blockText, &errOff)) {
            report.utf8ErrorCount++;
            report.corruptedTextCount++;
            report.rejectedBlocks++;
            addIssue(idx, blk.sourceFile, "corrupted_utf8", "Invalid UTF-8 at offset " + std::to_string(errOff));
            continue;
        }
        if (containsReplacementOrInvalidCtrl(blockText)) {
            report.corruptedTextCount++;
            report.rejectedBlocks++;
            addIssue(idx, blk.sourceFile, "corrupted_text", "Contains U+FFFD or invalid control bytes");
            continue;
        }

        // Symbol-ratio corruption check (> 40% bizarre symbols excluding standard punctuation)
        size_t alnumOrSpace = 0;
        size_t weirdChars = 0;
        for (unsigned char c : blockText) {
            if (std::isalnum(c) || std::isspace(c) ||
                c == '.' || c == ',' || c == '?' || c == '!' || c == ':' || c == ';' ||
                c == '\'' || c == '"' || c == '-' || c == '(' || c == ')' || c == '/' ||
                c == '+' || c == '=' || c == '%' || c == '*' || c == '_') {
                alnumOrSpace++;
            } else if (c < 0x80) {
                weirdChars++;
            }
        }
        if (blockText.size() > 8 && weirdChars * 3 > blockText.size()) {
            report.corruptedTextCount++;
            report.rejectedBlocks++;
            addIssue(idx, blk.sourceFile, "corrupted_text", "Excessive non-standard symbol ratio");
            continue;
        }

        bool blockRejected = false;
        std::string primaryQuestion;
        std::string primaryAnswer;

        // 2. Formatting, Conversation Structure, Incomplete Sentence, & Low-Info Checks
        if (blk.isInstructionFormat) {
            std::istringstream bStream(blockText);
            std::string line;
            std::vector<std::pair<std::string, std::string>> turns;
            bool badLine = false;

            while (std::getline(bStream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::string tLine = trimCopy(line);
                if (tLine.empty()) continue;

                if (tLine.rfind("User:", 0) == 0) {
                    turns.emplace_back("user", trimCopy(tLine.substr(5)));
                } else if (tLine.rfind("Assistant:", 0) == 0) {
                    turns.emplace_back("assistant", trimCopy(tLine.substr(10)));
                } else {
                    badLine = true;
                    break;
                }
            }

            if (badLine || turns.size() < 2) {
                report.formattingErrorCount++;
                report.rejectedBlocks++;
                addIssue(idx, blk.sourceFile, "formatting_error", "Missing User:/Assistant: prefixes or fewer than 2 turns");
                continue;
            }

            // Conversation alternation: must start with user, alternate, end with assistant
            bool convMalformed = false;
            if (turns.front().first != "user" || turns.back().first != "assistant") {
                convMalformed = true;
            }
            for (size_t t = 0; t < turns.size(); ++t) {
                if (turns[t].second.empty()) {
                    convMalformed = true;
                    break;
                }
                if (t > 0 && turns[t].first == turns[t - 1].first) {
                    convMalformed = true;
                    break;
                }
            }
            if (convMalformed) {
                report.malformedConversationCount++;
                report.rejectedBlocks++;
                addIssue(idx, blk.sourceFile, "malformed_conversation", "Invalid turn order, empty turn, or unclosed User turn");
                continue;
            }

            primaryQuestion = turns[0].second;
            primaryAnswer = turns[1].second;

            // Validate each turn content
            for (const auto& turn : turns) {
                auto words = splitWordsNormalized(turn.second);
                if (turn.first == "user") {
                    if (words.size() < 2) {
                        report.lowInformationCount++;
                        blockRejected = true;
                        addIssue(idx, blk.sourceFile, "low_information", "User prompt has fewer than 2 words");
                        break;
                    }
                } else {
                    const std::string& ans = turn.second;
                    std::string ansLower = toLowerCopy(ans);
                    // Placeholder detection
                    if (ansLower == "todo" || ansLower == "tbd" || ansLower == "n/a" ||
                        ansLower.find("lorem ipsum") != std::string::npos ||
                        ansLower.find("asdf") != std::string::npos) {
                        report.lowInformationCount++;
                        blockRejected = true;
                        addIssue(idx, blk.sourceFile, "low_information", "Contains placeholder text");
                        break;
                    }
                    if (words.size() < 4) {
                        report.lowInformationCount++;
                        blockRejected = true;
                        addIssue(idx, blk.sourceFile, "low_information", "Assistant answer has fewer than 4 words");
                        break;
                    }
                    // Incomplete sentence check: ellipsis truncation, dangling quote/paren, or missing end punctuation
                    char lastCh = ans.back();
                    bool endsCleanly = (lastCh == '.' || lastCh == '!' || lastCh == '?' || lastCh == '"' || lastCh == '\'' || lastCh == ')');
                    bool hasEllipsisCut = (ans.size() >= 3 && ans.substr(ans.size() - 3) == "...");
                    int parenBal = 0;
                    int doubleQuotes = 0;
                    for (char ch : ans) {
                        if (ch == '(') parenBal++;
                        else if (ch == ')') parenBal--;
                        else if (ch == '"') doubleQuotes++;
                    }
                    if (!endsCleanly || hasEllipsisCut || parenBal != 0 || (doubleQuotes % 2 != 0)) {
                        report.incompleteSentenceCount++;
                        blockRejected = true;
                        addIssue(idx, blk.sourceFile, "incomplete_sentence", "Assistant response ends abruptly or has unclosed quotes/parentheses");
                        break;
                    }
                }
            }
            if (blockRejected) {
                report.rejectedBlocks++;
                continue;
            }

            // Tautological echo check (first turn answer echoes question verbatim)
            std::string nQ = normalizeAlphaNumSpace(primaryQuestion);
            std::string nA = normalizeAlphaNumSpace(primaryAnswer);
            if (!nQ.empty() && nQ == nA) {
                report.lowInformationCount++;
                report.rejectedBlocks++;
                addIssue(idx, blk.sourceFile, "low_information", "Assistant answer merely echoes User prompt");
                continue;
            }
        } else {
            // Plain text prose block validation
            auto words = splitWordsNormalized(blockText);
            if (words.size() < 5) {
                report.lowInformationCount++;
                report.rejectedBlocks++;
                addIssue(idx, blk.sourceFile, "low_information", "Plain-text block fewer than 5 words");
                continue;
            }
            char lastCh = blockText.back();
            bool endsCleanly = (lastCh == '.' || lastCh == '!' || lastCh == '?' || lastCh == '"' || lastCh == ')');
            bool hasEllipsisCut = (blockText.size() >= 3 && blockText.substr(blockText.size() - 3) == "...");
            if (!endsCleanly || hasEllipsisCut) {
                report.incompleteSentenceCount++;
                report.rejectedBlocks++;
                addIssue(idx, blk.sourceFile, "incomplete_sentence", "Prose block lacks terminal punctuation");
                continue;
            }
            primaryQuestion = blockText;
            primaryAnswer = blockText;
        }

        // 3. Internal Repetition Check (consecutive duplicate words or high 3-gram repetition)
        auto blockWords = splitWordsNormalized(blockText);
        bool hasConsecutiveDup = false;
        for (size_t w = 1; w < blockWords.size(); ++w) {
            if (blockWords[w].size() >= 2 && blockWords[w] == blockWords[w - 1] && !std::isdigit(static_cast<unsigned char>(blockWords[w][0]))) {
                hasConsecutiveDup = true;
                break;
            }
        }
        double internalRepeatRatio = 0.0;
        if (blockWords.size() >= 8) {
            std::unordered_set<uint64_t> seenTrigrams;
            size_t totalTrigrams = blockWords.size() - 2;
            size_t dupTrigrams = 0;
            for (size_t w = 0; w < totalTrigrams; ++w) {
                // Exclude structural prefixes "user" / "assistant" from triggering multi-turn repetition
                if (blockWords[w] == "user" || blockWords[w] == "assistant") continue;
                uint64_t h = hashString64(blockWords[w] + "|" + blockWords[w + 1] + "|" + blockWords[w + 2]);
                if (!seenTrigrams.insert(h).second) dupTrigrams++;
            }
            internalRepeatRatio = totalTrigrams > 0 ? static_cast<double>(dupTrigrams) / static_cast<double>(totalTrigrams) : 0.0;
        }
        if (hasConsecutiveDup || internalRepeatRatio > 0.35) {
            report.repetitiveExampleCount++;
            report.rejectedBlocks++;
            addIssue(idx, blk.sourceFile, "repetitive_example", "Consecutive duplicate words or excessive internal trigram repetition");
            continue;
        }

        // 4. Exact & Normalized Duplicate Check
        std::string normWhole = normalizeAlphaNumSpace(blockText);
        if (!exactNormSeen.insert(normWhole).second) {
            report.exactDuplicatesRemoved++;
            report.rejectedBlocks++;
            addIssue(idx, blk.sourceFile, "exact_duplicate", "Normalized block matches existing example");
            continue;
        }

        // 5. Near-Duplicate Check (word 3-gram Jaccard >= nearDuplicateThreshold)
        std::vector<uint64_t> blkShingles = buildWordNgramHashes(blockWords, blockWords.size() >= 6 ? 3 : 2);
        std::unordered_set<size_t> candidateIndices;
        for (uint64_t h : blkShingles) {
            auto it = ngramToAccepted.find(h);
            if (it != ngramToAccepted.end()) {
                for (size_t cIdx : it->second) candidateIndices.insert(cIdx);
            }
        }
        bool isNearDup = false;
        for (size_t cIdx : candidateIndices) {
            float sim = computeSortedHashJaccard(blkShingles, acceptedMeta[cIdx].blockNgrams);
            if (sim >= nearDuplicateThreshold) {
                isNearDup = true;
                break;
            }
        }
        if (isNearDup) {
            report.nearDuplicatesRemoved++;
            report.rejectedBlocks++;
            addIssue(idx, blk.sourceFile, "near_duplicate", "High shingle Jaccard similarity with an accepted example");
            continue;
        }

        // 6. Contradictory Answer Check (same question -> conflicting answer)
        std::vector<uint64_t> qShingles;
        if (blk.isInstructionFormat && !primaryQuestion.empty()) {
            auto qWords = splitWordsNormalized(primaryQuestion);
            qShingles = buildWordNgramHashes(qWords, qWords.size() >= 4 ? 2 : 1);
            std::unordered_set<size_t> qCandidates;
            for (uint64_t h : qShingles) {
                auto it = qNgramToAccepted.find(h);
                if (it != qNgramToAccepted.end()) {
                    for (size_t cIdx : it->second) qCandidates.insert(cIdx);
                }
            }
            bool contradicts = false;
            for (size_t cIdx : qCandidates) {
                float qSim = computeSortedHashJaccard(qShingles, acceptedMeta[cIdx].qNgrams);
                if (qSim >= 0.90f) {
                    auto ansWordsNew = splitWordsNormalized(primaryAnswer);
                    auto ansWordsOld = splitWordsNormalized(acceptedMeta[cIdx].firstAnswer);
                    int polNew = 0, polOld = 0;
                    bool hasPolNew = hasYesNoPolarity(ansWordsNew, polNew);
                    bool hasPolOld = hasYesNoPolarity(ansWordsOld, polOld);
                    if (hasPolNew && hasPolOld && polNew != polOld) {
                        contradicts = true;
                    }
                    auto numsNew = extractNumbers(primaryAnswer);
                    auto numsOld = extractNumbers(acceptedMeta[cIdx].firstAnswer);
                    if (!numsNew.empty() && !numsOld.empty() && numsNew != numsOld) {
                        contradicts = true;
                    }
                    auto ansSetNew = buildWordNgramHashes(ansWordsNew, 1);
                    auto ansSetOld = buildWordNgramHashes(ansWordsOld, 1);
                    float aSim = computeSortedHashJaccard(ansSetNew, ansSetOld);
                    if (aSim < 0.20f) {
                        contradicts = true;
                    }
                    if (contradicts) break;
                }
            }
            if (contradicts) {
                report.contradictoryAnswerCount++;
                report.rejectedBlocks++;
                addIssue(idx, blk.sourceFile, "contradictory_answer", "Conflicting answer for nearly identical question");
                continue;
            }
        }

        // Accepted! (rejected blocks have already skipped via continue; if !filterFailed, caller may collect all valid blocks)
        (void)filterFailed;
        size_t accIdx = outAcceptedBlocks.size();
        outAcceptedBlocks.push_back(blk);
        AcceptedRecord rec;
        rec.blockNgrams = std::move(blkShingles);
        rec.qNgrams = std::move(qShingles);
        rec.firstQuestion = primaryQuestion;
        rec.firstAnswer = primaryAnswer;
        for (uint64_t h : rec.blockNgrams) ngramToAccepted[h].push_back(accIdx);
        for (uint64_t h : rec.qNgrams) qNgramToAccepted[h].push_back(accIdx);
        acceptedMeta.push_back(std::move(rec));
    }

    report.acceptedBlocks = static_cast<uint32_t>(outAcceptedBlocks.size());
    uint32_t totalDup = report.exactDuplicatesRemoved + report.nearDuplicatesRemoved;
    report.duplicationRate = report.totalInputBlocks > 0
        ? static_cast<float>(totalDup) / static_cast<float>(report.totalInputBlocks)
        : 0.0f;
    report.passedQualityGate = (report.acceptedBlocks > 0 && report.utf8ErrorCount == 0 && report.contaminationCount == 0);
    return report;
}

std::vector<ContaminationMatch> DatasetPipeline::checkContamination(
    const std::vector<std::pair<std::string, std::string>>& sourceDocs,
    const std::vector<UnseenTestQuestion>& unseenQuestions,
    float jaccardThreshold
) {
    std::vector<ContaminationMatch> matches;

    struct NormDoc {
        std::string source;
        std::string lowerRaw;
        std::string normAlpha;
        std::vector<std::vector<uint64_t>> lineShingles;
        std::vector<std::vector<uint64_t>> lineBigrams;
    };
    std::vector<NormDoc> docs;
    docs.reserve(sourceDocs.size());

    for (const auto& sd : sourceDocs) {
        NormDoc nd;
        nd.source = sd.first;
        nd.lowerRaw = toLowerCopy(sd.second);
        nd.normAlpha = normalizeAlphaNumSpace(sd.second);
        std::istringstream iss(sd.second);
        std::string line;
        while (std::getline(iss, line)) {
            std::string trimmed = trimCopy(line);
            if (trimmed.empty()) continue;

            // If JSONL record, unpack into User:/Assistant: lines before splitting
            std::string expanded = trimmed;
            if (trimmed.front() == '{') {
                std::string parsedBlock, src, lic;
                if (parseJsonlInstructionLine(trimmed, parsedBlock, src, lic, nullptr)) {
                    expanded = parsedBlock;
                    // Also append lowercase parsed text for exact substring match
                    nd.lowerRaw += "\n" + toLowerCopy(parsedBlock);
                    nd.normAlpha += " " + normalizeAlphaNumSpace(parsedBlock);
                }
            }

            std::istringstream subIss(expanded);
            std::string subLine;
            while (std::getline(subIss, subLine)) {
                std::string tSub = trimCopy(subLine);
                if (tSub.rfind("User:", 0) == 0) tSub = trimCopy(tSub.substr(5));
                else if (tSub.rfind("Assistant:", 0) == 0) tSub = trimCopy(tSub.substr(10));
                if (tSub.empty()) continue;
                auto words = splitWordsNormalized(tSub);
                if (words.size() >= 4) {
                    nd.lineShingles.push_back(buildWordNgramHashes(words, words.size() >= 6 ? 3 : 2));
                    nd.lineBigrams.push_back(buildWordNgramHashes(words, words.size() >= 3 ? 2 : 1));
                }
            }
        }
        docs.push_back(std::move(nd));
    }

    for (const auto& uq : unseenQuestions) {
        std::string qLower = toLowerCopy(uq.question);
        std::string qNorm = normalizeAlphaNumSpace(uq.question);

        // For multi-turn unseen prompts ("User: ... \nAssistant: ... \nUser: finalQ"), check final turn as well
        std::string finalTurn = uq.question;
        size_t lastUser = finalTurn.rfind("User:");
        if (lastUser != std::string::npos) {
            finalTurn = trimCopy(finalTurn.substr(lastUser + 5));
        }
        auto qWords = splitWordsNormalized(finalTurn);
        auto qShingles = buildWordNgramHashes(qWords, qWords.size() >= 6 ? 3 : 2);
        auto qBigrams = buildWordNgramHashes(qWords, qWords.size() >= 3 ? 2 : 1);

        bool matchedThisQ = false;
        for (const auto& nd : docs) {
            if (!qLower.empty() && nd.lowerRaw.find(qLower) != std::string::npos) {
                matches.push_back({uq.question, nd.source, "exact_substring", 1.0f});
                matchedThisQ = true;
                break;
            }
            if (!qNorm.empty() && qNorm.size() >= 12 && nd.normAlpha.find(qNorm) != std::string::npos) {
                matches.push_back({uq.question, nd.source, "normalized_match", 1.0f});
                matchedThisQ = true;
                break;
            }
            if (!qShingles.empty()) {
                for (size_t l = 0; l < nd.lineShingles.size(); ++l) {
                    float sim = computeSortedHashJaccard(qShingles, nd.lineShingles[l]);
                    float biSim = (qWords.size() >= 5 && !qBigrams.empty() && !nd.lineBigrams[l].empty())
                        ? computeSortedHashJaccard(qBigrams, nd.lineBigrams[l])
                        : 0.0f;
                    if (sim >= jaccardThreshold || biSim >= 0.60f) {
                        matches.push_back({uq.question, nd.source, "ngram_jaccard", std::max(sim, biSim)});
                        matchedThisQ = true;
                        break;
                    }
                }
                if (matchedThisQ) break;
            }
        }
    }

    return matches;
}

bool DatasetPipeline::loadFromFile(
    const std::string& filePath,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen
) {
    return loadFromFiles({filePath}, tokenizer, trainValSplit, seqLen, false);
}

bool DatasetPipeline::loadFromFiles(
    const std::vector<std::string>& filePaths,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen,
    bool deduplicate
) {
    if (filePaths.empty()) {
        metadata_.isValid = false;
        metadata_.errorMessage = "No file paths provided to loadFromFiles";
        return false;
    }

    metadata_ = DatasetMetadata();
    metadata_.splitRatio = trainValSplit;
    metadata_.seqLen = seqLen;
    metadata_.sourceFiles = filePaths;
    metadata_.documentsCount = static_cast<uint32_t>(filePaths.size());
    metadata_.source = "unspecified";
    metadata_.license = "UNVERIFIED";
    metadata_.licenseVerified = false;

    std::unordered_set<std::string> seenLines;
    std::string combinedText;
    uint64_t totalBytes = 0;
    uint32_t uniqueCount = 0;
    uint32_t dupCount = 0;

    for (const auto& path : filePaths) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            metadata_.isValid = false;
            metadata_.errorMessage = "Failed to open dataset file: " + path;
            return false;
        }

        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size <= 0) continue;
        totalBytes += static_cast<uint64_t>(size);

        std::string content(static_cast<size_t>(size), '\0');
        file.read(&content[0], size);
        file.close();

        // Strict UTF-8 validation
        size_t errOffset = 0;
        if (!validateUtf8(content, &errOffset)) {
            metadata_.isValid = false;
            metadata_.errorMessage = "Invalid UTF-8 in file " + path + " at byte offset " + std::to_string(errOffset);
            return false;
        }

        if (deduplicate) {
            std::istringstream stream(content);
            std::string line;
            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();

                // If blank line, keep single newline for structure
                if (line.empty()) {
                    combinedText += "\n";
                    continue;
                }

                if (seenLines.find(line) != seenLines.end()) {
                    dupCount++;
                    continue;
                }

                seenLines.insert(line);
                uniqueCount++;
                combinedText += line + "\n";
            }
        } else {
            combinedText += content + "\n";
        }
    }

    metadata_.fileSizeBytes = totalBytes;
    metadata_.uniqueLines = uniqueCount;
    metadata_.duplicatesRemoved = dupCount;
    metadata_.exactDuplicatesRemoved = dupCount;
    metadata_.contentHashHex = computeContentHashHex(combinedText);
    metadata_.datasetVersion = "v-" + metadata_.contentHashHex.substr(0, 8);

    std::string compositeName = filePaths.size() == 1 ? filePaths[0] : (std::to_string(filePaths.size()) + " files");
    return loadFromString(combinedText, compositeName, tokenizer, trainValSplit, seqLen);
}

bool DatasetPipeline::loadFromDirectory(
    const std::string& dirPath,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen,
    bool deduplicate
) {
    namespace fs = std::filesystem;
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        metadata_.isValid = false;
        metadata_.errorMessage = "Directory does not exist: " + dirPath;
        return false;
    }

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());
    if (files.empty()) {
        metadata_.isValid = false;
        metadata_.errorMessage = "No .txt files found in directory: " + dirPath;
        return false;
    }

    return loadFromFiles(files, tokenizer, trainValSplit, seqLen, deduplicate);
}

bool DatasetPipeline::loadFromString(
    const std::string& text,
    const std::string& datasetName,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen
) {
    metadata_.filePath = datasetName;
    metadata_.splitRatio = trainValSplit;
    metadata_.seqLen = seqLen;
    metadata_.totalCharacters = text.size();
    if (metadata_.contentHashHex.empty()) {
        metadata_.contentHashHex = computeContentHashHex(text);
        metadata_.datasetVersion = "v-" + metadata_.contentHashHex.substr(0, 8);
    }

    // Check non-empty and non-whitespace
    bool hasNonWhitespace = false;
    for (char c : text) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            hasNonWhitespace = true;
            break;
        }
    }

    if (!hasNonWhitespace) {
        metadata_.isValid = false;
        metadata_.errorMessage = "Dataset contains only whitespace or is empty";
        return false;
    }

    // Strict UTF-8 validation
    size_t errOffset = 0;
    if (!validateUtf8(text, &errOffset)) {
        metadata_.isValid = false;
        metadata_.errorMessage = "Corrupted dataset: Invalid UTF-8 byte sequence at offset " + std::to_string(errOffset);
        return false;
    }

    // Tokenize
    allTokens_ = tokenizer.encode(text, false, false);
    metadata_.totalTokens = allTokens_.size();

    std::unordered_set<int32_t> uniqToks(allTokens_.begin(), allTokens_.end());
    metadata_.uniqueTokensCount = uniqToks.size();

    if (allTokens_.size() < 4) {
        metadata_.isValid = false;
        metadata_.errorMessage = "Dataset is too small to construct training sequences (fewer than 4 tokens)";
        return false;
    }

    // Split into train / val
    size_t splitPoint = static_cast<size_t>(allTokens_.size() * trainValSplit);
    if (splitPoint < 2) splitPoint = 2;
    if (splitPoint >= allTokens_.size()) splitPoint = allTokens_.size() - 1;

    trainTokens_.assign(allTokens_.begin(), allTokens_.begin() + splitPoint);
    valTokens_.assign(allTokens_.begin() + splitPoint, allTokens_.end());

    metadata_.trainTokensCount = trainTokens_.size();
    metadata_.valTokensCount = valTokens_.size();

    // Build sliding window sequences
    buildSequences(seqLen);

    metadata_.trainSequencesCount = static_cast<uint32_t>(trainSequences_.size());
    metadata_.valSequencesCount = static_cast<uint32_t>(valSequences_.size());
    metadata_.isValid = true;
    metadata_.errorMessage.clear();

    return true;
}

void DatasetPipeline::buildSequences(uint32_t seqLen) {
    trainSequences_.clear();
    valSequences_.clear();

    // Helper to generate chunks with 50% stride
    auto makeChunks = [seqLen](const std::vector<int32_t>& tokens, std::vector<SequenceSample>& out) {
        if (tokens.size() < seqLen + 1) {
            std::vector<int32_t> padded = tokens;
            while (padded.size() < seqLen + 1) {
                for (int32_t t : tokens) {
                    padded.push_back(t);
                    if (padded.size() >= seqLen + 1) break;
                }
            }
            SequenceSample sample;
            sample.inputs.assign(padded.begin(), padded.begin() + seqLen);
            sample.targets.assign(padded.begin() + 1, padded.begin() + seqLen + 1);
            out.push_back(sample);
            return;
        }

        size_t stride = std::max(size_t(1), size_t(seqLen / 2));
        for (size_t offset = 0; offset + seqLen + 1 <= tokens.size(); offset += stride) {
            SequenceSample sample;
            sample.inputs.assign(tokens.begin() + offset, tokens.begin() + offset + seqLen);
            sample.targets.assign(tokens.begin() + offset + 1, tokens.begin() + offset + seqLen + 1);
            out.push_back(sample);
        }
    };

    makeChunks(trainTokens_, trainSequences_);
    makeChunks(valTokens_, valSequences_);
}

bool DatasetPipeline::loadInstructionExamplesFromFiles(
    const std::vector<std::string>& filePaths,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen,
    bool maskPrompts,
    uint64_t splitSeed,
    const std::string& source,
    const std::string& license,
    bool enforceQualityChecks,
    const std::string& unseenTestPath
) {
    if (filePaths.empty()) {
        metadata_.isValid = false;
        metadata_.errorMessage = "No file paths provided to loadInstructionExamplesFromFiles";
        return false;
    }

    metadata_ = DatasetMetadata();
    qualityReport_ = DatasetQualityReport();
    metadata_.filePath = filePaths.size() == 1 ? filePaths[0] : (std::to_string(filePaths.size()) + " files");
    metadata_.sourceFiles = filePaths;
    metadata_.documentsCount = static_cast<uint32_t>(filePaths.size());
    metadata_.source = source.empty() ? "unspecified" : source;
    metadata_.license = license.empty() ? "UNVERIFIED" : license;
    metadata_.licenseVerified = isVerifiedPermissiveLicense(metadata_.license);
    metadata_.splitRatio = trainValSplit;
    metadata_.splitSeed = splitSeed;
    metadata_.seqLen = seqLen;

    allTokens_.clear();
    trainTokens_.clear();
    valTokens_.clear();
    trainSequences_.clear();
    valSequences_.clear();
    acceptedExampleTexts_.clear();

    std::vector<RawExampleBlock> rawBlocks;
    std::vector<std::pair<std::string, std::string>> sourceDocs;
    uint64_t totalBytes = 0;
    uint32_t invalidCount = 0;
    uint32_t jsonlParseErrors = 0;

    for (const auto& path : filePaths) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            metadata_.isValid = false;
            metadata_.errorMessage = "Failed to open dataset file: " + path;
            return false;
        }
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size <= 0) continue;

        totalBytes += static_cast<uint64_t>(size);
        std::string content(static_cast<size_t>(size), '\0');
        file.read(&content[0], size);
        file.close();

        size_t errOffset = 0;
        if (!validateUtf8(content, &errOffset)) {
            metadata_.isValid = false;
            metadata_.errorMessage = "Invalid UTF-8 in file " + path + " at byte offset " + std::to_string(errOffset);
            return false;
        }

        sourceDocs.emplace_back(path, content);
        bool isJsonl = (path.size() >= 6 && path.substr(path.size() - 6) == ".jsonl");

        DatasetSourceRecord srcRec;
        srcRec.path = path;
        srcRec.id = std::filesystem::path(path).filename().string();
        srcRec.format = isJsonl ? "instruction_jsonl" : "instruction_txt";
        srcRec.source = metadata_.source;
        srcRec.license = metadata_.license;
        srcRec.licenseVerified = metadata_.licenseVerified;
        srcRec.bytes = static_cast<uint64_t>(size);
        srcRec.documents = 1;

        if (isJsonl) {
            std::istringstream stream(content);
            std::string line;
            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::string tLine = trimCopy(line);
                if (tLine.empty()) continue;

                std::string blkText, lineSrc, lineLic, parseErr;
                if (!parseJsonlInstructionLine(tLine, blkText, lineSrc, lineLic, &parseErr)) {
                    jsonlParseErrors++;
                    invalidCount++;
                    continue;
                }
                if (lineLic != "UNVERIFIED" && metadata_.license == "UNVERIFIED") {
                    // Track per-record provenance without silently elevating unverified files
                    srcRec.license = lineLic;
                    srcRec.licenseVerified = isVerifiedPermissiveLicense(lineLic);
                }
                RawExampleBlock rb;
                rb.text = std::move(blkText);
                rb.sourceFile = path;
                rb.source = (lineSrc != "unspecified") ? lineSrc : metadata_.source;
                rb.license = (lineLic != "UNVERIFIED") ? lineLic : metadata_.license;
                rb.isInstructionFormat = true;
                rawBlocks.push_back(std::move(rb));
                srcRec.examples++;
            }
        } else {
            std::istringstream stream(content);
            std::string line;
            std::string currentBlock;
            auto flushBlock = [&]() {
                while (!currentBlock.empty() && currentBlock.back() == '\n') {
                    currentBlock.pop_back();
                }
                if (currentBlock.empty()) return;
                bool hasInstructionTag = (currentBlock.find("User:") != std::string::npos &&
                                          currentBlock.find("Assistant:") != std::string::npos);
                RawExampleBlock rb;
                rb.text = currentBlock;
                rb.sourceFile = path;
                rb.source = metadata_.source;
                rb.license = metadata_.license;
                rb.isInstructionFormat = hasInstructionTag;
                rawBlocks.push_back(std::move(rb));
                srcRec.examples++;
                currentBlock.clear();
            };

            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();

                bool onlySpace = true;
                for (char c : line) {
                    if (!std::isspace(static_cast<unsigned char>(c))) {
                        onlySpace = false;
                        break;
                    }
                }
                if (onlySpace) {
                    flushBlock();
                    continue;
                }

                bool hasInvalidCtrl = false;
                for (unsigned char c : line) {
                    if (c < 0x20 && c != '\t') {
                        hasInvalidCtrl = true;
                        break;
                    }
                }
                if (hasInvalidCtrl) {
                    invalidCount++;
                    continue;
                }

                currentBlock += line + "\n";
            }
            flushBlock();
        }
        metadata_.sourceRecords.push_back(std::move(srcRec));
    }

    if (rawBlocks.empty()) {
        metadata_.isValid = false;
        metadata_.errorMessage = "No valid example blocks found in dataset";
        return false;
    }

    std::vector<RawExampleBlock> acceptedBlocks;
    if (enforceQualityChecks) {
        qualityReport_ = analyzeAndFilterBlocks(rawBlocks, acceptedBlocks, 0.85f, true);
        qualityReport_.formattingErrorCount += jsonlParseErrors;
    } else {
        // Legacy exact-block deduplication compatibility mode
        std::unordered_set<std::string> seenBlocks;
        uint32_t dupCount = 0;
        for (const auto& rb : rawBlocks) {
            if (seenBlocks.find(rb.text) != seenBlocks.end()) {
                dupCount++;
            } else {
                seenBlocks.insert(rb.text);
                acceptedBlocks.push_back(rb);
            }
        }
        qualityReport_.totalInputBlocks = static_cast<uint32_t>(rawBlocks.size());
        qualityReport_.acceptedBlocks = static_cast<uint32_t>(acceptedBlocks.size());
        qualityReport_.exactDuplicatesRemoved = dupCount;
        qualityReport_.rejectedBlocks = dupCount;
        qualityReport_.passedQualityGate = !acceptedBlocks.empty();
    }

    if (acceptedBlocks.empty()) {
        metadata_.isValid = false;
        metadata_.errorMessage = "All dataset blocks were rejected by quality filtering";
        return false;
    }

    // Contamination check against held-out unseen evaluation questions if path provided
    if (!unseenTestPath.empty() && std::filesystem::exists(unseenTestPath)) {
        std::vector<UnseenTestQuestion> unseenQs;
        std::string uErr;
        if (loadUnseenQuestions(unseenTestPath, unseenQs, &uErr)) {
            std::vector<std::pair<std::string, std::string>> acceptedDocs;
            for (const auto& ab : acceptedBlocks) {
                acceptedDocs.emplace_back(ab.sourceFile, ab.text);
            }
            auto matches = checkContamination(acceptedDocs, unseenQs, 0.72f);
            qualityReport_.contaminationCount = static_cast<uint32_t>(matches.size());
            metadata_.contaminationCount = qualityReport_.contaminationCount;
            if (!matches.empty()) {
                qualityReport_.passedQualityGate = false;
            }
        }
    }

    // Shuffle examples deterministically before train/val split so every domain is balanced
    std::vector<size_t> idxOrder(acceptedBlocks.size());
    for (size_t i = 0; i < idxOrder.size(); ++i) idxOrder[i] = i;
    std::mt19937_64 rng(splitSeed);
    std::shuffle(idxOrder.begin(), idxOrder.end(), rng);

    std::vector<RawExampleBlock> shuffledBlocks;
    shuffledBlocks.reserve(acceptedBlocks.size());
    std::string combinedHashInput;
    uint64_t totalChars = 0;
    uint32_t uniqueLinesCount = 0;

    for (size_t i : idxOrder) {
        shuffledBlocks.push_back(acceptedBlocks[i]);
        acceptedExampleTexts_.push_back(acceptedBlocks[i].text);
        totalChars += acceptedBlocks[i].text.size();
        combinedHashInput += acceptedBlocks[i].text + "\n";
        std::istringstream ls(acceptedBlocks[i].text);
        std::string l;
        while (std::getline(ls, l)) {
            if (!trimCopy(l).empty()) uniqueLinesCount++;
        }
    }

    size_t splitIdx = static_cast<size_t>(shuffledBlocks.size() * trainValSplit);
    if (splitIdx < 1) splitIdx = 1;
    if (splitIdx >= shuffledBlocks.size() && shuffledBlocks.size() >= 2) splitIdx = shuffledBlocks.size() - 1;

    metadata_.fileSizeBytes = totalBytes;
    metadata_.totalCharacters = totalChars;
    metadata_.uniqueLines = uniqueLinesCount;
    metadata_.exactDuplicatesRemoved = qualityReport_.exactDuplicatesRemoved;
    metadata_.nearDuplicatesRemoved = qualityReport_.nearDuplicatesRemoved;
    metadata_.duplicatesRemoved = qualityReport_.exactDuplicatesRemoved + qualityReport_.nearDuplicatesRemoved;
    metadata_.qualityIssuesRemoved = qualityReport_.rejectedBlocks;
    metadata_.invalidLinesRemoved = invalidCount;
    metadata_.exampleCount = static_cast<uint32_t>(shuffledBlocks.size());
    metadata_.trainExamplesCount = static_cast<uint32_t>(splitIdx);
    metadata_.valExamplesCount = static_cast<uint32_t>(shuffledBlocks.size() - splitIdx);
    metadata_.contentHashHex = computeContentHashHex(combinedHashInput);
    metadata_.datasetVersion = "v-" + metadata_.contentHashHex.substr(0, 8) + "-seed" + std::to_string(splitSeed);

    auto encodeBlockToSamples = [&](const RawExampleBlock& rb, bool isTrain) {
        const std::string& block = rb.text;
        std::vector<int32_t> tokens;
        std::vector<bool> trainMask;

        tokens.push_back(tokenizer.specialTokens().bosId);
        trainMask.push_back(false);

        std::vector<size_t> turnStartTokenIndices;
        turnStartTokenIndices.push_back(0); // Turn 0 starts cleanly at BOS

        std::istringstream bStream(block);
        std::string line;
        bool isFirstUserTurn = true;
        while (std::getline(bStream, line)) {
            if (line.empty()) continue;
            if (maskPrompts && rb.isInstructionFormat && line.rfind("User:", 0) == 0) {
                if (!isFirstUserTurn) {
                    turnStartTokenIndices.push_back(tokens.size());
                }
                isFirstUserTurn = false;
                auto enc = tokenizer.encode(line + "\n", false, false);
                for (int32_t t : enc) {
                    tokens.push_back(t);
                    trainMask.push_back(false);
                }
            } else if (maskPrompts && rb.isInstructionFormat && line.rfind("Assistant:", 0) == 0) {
                const std::string prefix = "Assistant:";
                auto pEnc = tokenizer.encode(prefix, false, false);
                for (int32_t t : pEnc) {
                    tokens.push_back(t);
                    trainMask.push_back(false);
                }
                std::string ansPart = line.substr(prefix.size()) + "\n";
                auto aEnc = tokenizer.encode(ansPart, false, false);
                for (int32_t t : aEnc) {
                    tokens.push_back(t);
                    trainMask.push_back(true);
                }
            } else {
                auto enc = tokenizer.encode(line + "\n", false, false);
                for (int32_t t : enc) {
                    tokens.push_back(t);
                    trainMask.push_back(true);
                }
            }
        }

        // End of example EOS token (unmasked so model learns when to stop)
        tokens.push_back(tokenizer.specialTokens().eosId);
        trainMask.push_back(true);

        for (int32_t t : tokens) {
            allTokens_.push_back(t);
            if (isTrain) trainTokens_.push_back(t);
            else valTokens_.push_back(t);
        }

        if (tokens.size() < 2) return;

        size_t totalPairs = tokens.size() - 1;
        size_t stepStride = std::max(size_t(1), static_cast<size_t>(seqLen / 2));

        auto emitWindow = [&](size_t start, size_t len) {
            if (len == 0) return;
            SequenceSample s;
            s.inputs.reserve(len);
            s.targets.reserve(len);
            uint64_t unmaskedHere = 0;
            uint64_t maskedHere = 0;
            for (size_t k = 0; k < len; ++k) {
                s.inputs.push_back(tokens[start + k]);
                bool predict = trainMask[start + k + 1];
                if (predict) {
                    s.targets.push_back(tokens[start + k + 1]);
                    unmaskedHere++;
                } else {
                    s.targets.push_back(IGNORE_INDEX);
                    maskedHere++;
                }
            }
            if (unmaskedHere > 0) {
                metadata_.unmaskedTargetTokensCount += unmaskedHere;
                metadata_.maskedTokensCount += maskedHere;
                if (isTrain) trainSequences_.push_back(std::move(s));
                else valSequences_.push_back(std::move(s));
            }
        };

        if (totalPairs <= seqLen) {
            emitWindow(0, totalPairs);
        } else if (rb.isInstructionFormat && turnStartTokenIndices.size() > 1) {
            // Align windows to clean User: turn starts rather than cutting mid-Assistant answer
            size_t maxCoveredEnd = 0;
            for (size_t tIdx = 0; tIdx < turnStartTokenIndices.size(); ++tIdx) {
                size_t tStart = turnStartTokenIndices[tIdx];
                if (tStart >= totalPairs) break;
                size_t nextTurnStart = (tIdx + 1 < turnStartTokenIndices.size())
                    ? std::min(totalPairs, turnStartTokenIndices[tIdx + 1])
                    : totalPairs;
                if (tStart + seqLen <= maxCoveredEnd && nextTurnStart <= maxCoveredEnd) {
                    continue; // Already fully contained in the previous multi-turn window
                }
                if (nextTurnStart - tStart <= seqLen) {
                    size_t len = std::min(static_cast<size_t>(seqLen), totalPairs - tStart);
                    emitWindow(tStart, len);
                    maxCoveredEnd = std::max(maxCoveredEnd, tStart + len);
                } else {
                    // Single very long turn: step within this turn starting from its User: header
                    size_t pos = tStart;
                    while (pos < nextTurnStart) {
                        size_t len = std::min(static_cast<size_t>(seqLen), totalPairs - pos);
                        emitWindow(pos, len);
                        maxCoveredEnd = std::max(maxCoveredEnd, pos + len);
                        if (pos + len >= nextTurnStart) break;
                        pos += stepStride;
                    }
                }
            }
        } else {
            size_t pos = 0;
            while (pos < totalPairs) {
                size_t len = std::min(static_cast<size_t>(seqLen), totalPairs - pos);
                emitWindow(pos, len);
                if (pos + len >= totalPairs) break;
                pos += stepStride;
            }
        }
    };

    for (size_t i = 0; i < shuffledBlocks.size(); ++i) {
        encodeBlockToSamples(shuffledBlocks[i], i < splitIdx);
    }

    std::unordered_set<int32_t> uniqueTokIds(allTokens_.begin(), allTokens_.end());
    metadata_.totalTokens = allTokens_.size();
    metadata_.uniqueTokensCount = uniqueTokIds.size();
    metadata_.trainTokensCount = trainTokens_.size();
    metadata_.valTokensCount = valTokens_.size();
    metadata_.trainSequencesCount = static_cast<uint32_t>(trainSequences_.size());
    metadata_.valSequencesCount = static_cast<uint32_t>(valSequences_.size());
    metadata_.avgTokensPerExample = metadata_.exampleCount > 0
        ? static_cast<float>(metadata_.totalTokens) / static_cast<float>(metadata_.exampleCount)
        : 0.0f;
    metadata_.isValid = (!trainSequences_.empty() && !valSequences_.empty());
    if (!metadata_.isValid) {
        metadata_.errorMessage = "Failed to produce both train and validation instruction sequences";
    }
    return metadata_.isValid;
}

bool DatasetPipeline::loadInstructionExamplesFromDirectory(
    const std::string& dirPath,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen,
    bool maskPrompts,
    uint64_t splitSeed,
    const std::string& source,
    const std::string& license,
    bool enforceQualityChecks,
    const std::string& unseenTestPath
) {
    namespace fs = std::filesystem;
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        metadata_.isValid = false;
        metadata_.errorMessage = "Directory does not exist: " + dirPath;
        return false;
    }

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension();
            if (ext == ".txt" || ext == ".jsonl") {
                files.push_back(entry.path().string());
            }
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        metadata_.isValid = false;
        metadata_.errorMessage = "No .txt or .jsonl files found in directory: " + dirPath;
        return false;
    }

    return loadInstructionExamplesFromFiles(
        files, tokenizer, trainValSplit, seqLen, maskPrompts, splitSeed,
        source, license, enforceQualityChecks, unseenTestPath
    );
}

bool DatasetPipeline::loadManagedDataset(
    const std::string& datasetDirOrManifest,
    Tokenizer& tokenizer,
    float trainValSplit,
    uint32_t seqLen,
    bool maskPrompts,
    uint64_t splitSeed,
    const std::string& unseenTestPath,
    bool requireVerifiedLicense,
    bool rejectIfContaminated
) {
    namespace fs = std::filesystem;
    std::string targetDir = datasetDirOrManifest;
    std::string manifestPath;

    if (fs::is_regular_file(datasetDirOrManifest) &&
        fs::path(datasetDirOrManifest).extension() == ".json") {
        manifestPath = datasetDirOrManifest;
        targetDir = fs::path(datasetDirOrManifest).parent_path().string();
    } else if (fs::is_directory(datasetDirOrManifest)) {
        std::string candidate = (fs::path(datasetDirOrManifest) / "DATASET_MANIFEST.json").string();
        if (fs::exists(candidate)) {
            manifestPath = candidate;
        } else {
            std::string parentCand = (fs::path(datasetDirOrManifest).parent_path() / "DATASET_MANIFEST.json").string();
            if (fs::exists(parentCand)) {
                manifestPath = parentCand;
            }
        }
    } else {
        metadata_.isValid = false;
        metadata_.errorMessage = "Managed dataset path does not exist: " + datasetDirOrManifest;
        return false;
    }

    std::string declaredSource = "unspecified";
    std::string declaredLicense = "UNVERIFIED";
    std::string declaredVersion = "";

    if (!manifestPath.empty()) {
        std::ifstream mfs(manifestPath, std::ios::binary);
        if (mfs.is_open()) {
            std::string mContent((std::istreambuf_iterator<char>(mfs)), std::istreambuf_iterator<char>());
            std::string val;
            if (extractJsonTopLevelStringField(mContent, "source", val) && !trimCopy(val).empty()) {
                declaredSource = trimCopy(val);
            }
            if (extractJsonTopLevelStringField(mContent, "license", val) && !trimCopy(val).empty()) {
                declaredLicense = trimCopy(val);
            }
            if (extractJsonTopLevelStringField(mContent, "version", val) && !trimCopy(val).empty()) {
                declaredVersion = trimCopy(val);
            }
        }
    }

    if (requireVerifiedLicense && !isVerifiedPermissiveLicense(declaredLicense)) {
        metadata_ = DatasetMetadata();
        metadata_.filePath = datasetDirOrManifest;
        metadata_.source = declaredSource;
        metadata_.license = declaredLicense;
        metadata_.licenseVerified = false;
        metadata_.isValid = false;
        metadata_.errorMessage = "License verification failed: declared license \"" + declaredLicense + "\" is unverified or unknown (never assumed permissive)";
        return false;
    }

    std::string trainSubdir = (fs::path(targetDir) / "train_val").string();
    std::string loadDir = fs::is_directory(trainSubdir) && fs::is_directory(datasetDirOrManifest) &&
                          fs::path(datasetDirOrManifest).filename() != "train_val" &&
                          fs::is_regular_file(manifestPath) && fs::path(manifestPath).parent_path() == fs::path(targetDir)
                          ? trainSubdir
                          : targetDir;

    if (!fs::is_directory(loadDir)) loadDir = targetDir;

    bool ok = loadInstructionExamplesFromDirectory(
        loadDir, tokenizer, trainValSplit, seqLen, maskPrompts, splitSeed,
        declaredSource, declaredLicense, true, unseenTestPath
    );

    if (!ok) return false;

    if (!declaredVersion.empty()) {
        metadata_.datasetVersion = declaredVersion + "+" + metadata_.contentHashHex.substr(0, 8);
    }

    if (rejectIfContaminated && metadata_.contaminationCount > 0) {
        metadata_.isValid = false;
        metadata_.errorMessage = "Evaluation set contamination detected (" +
                                 std::to_string(metadata_.contaminationCount) +
                                 " overlapping prompt/n-gram matches with unseen test set)";
        return false;
    }

    return true;
}

bool DatasetPipeline::loadUnseenQuestions(
    const std::string& filePath,
    std::vector<UnseenTestQuestion>& outQuestions,
    std::string* outError
) {
    outQuestions.clear();
    std::ifstream ifs(filePath, std::ios::binary);
    if (!ifs.is_open()) {
        if (outError) *outError = "Cannot open unseen test question file: " + filePath;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    size_t errOffset = 0;
    if (!validateUtf8(content, &errOffset)) {
        if (outError) *outError = "Invalid UTF-8 in unseen test question file at offset " + std::to_string(errOffset);
        return false;
    }

    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        // Format: CATEGORY|EXPECTED_BEHAVIOR|kw1,kw2,...|QUESTION
        std::vector<std::string> parts;
        size_t start = 0;
        for (int p = 0; p < 3; ++p) {
            size_t pos = line.find('|', start);
            if (pos == std::string::npos) break;
            parts.push_back(line.substr(start, pos - start));
            start = pos + 1;
        }
        parts.push_back(line.substr(start));

        if (parts.size() != 4) continue;

        UnseenTestQuestion q;
        q.category = parts[0];
        q.expectedBehavior = parts[1];

        std::istringstream kwStream(parts[2]);
        std::string kw;
        while (std::getline(kwStream, kw, ',')) {
            if (!kw.empty()) q.keywords.push_back(kw);
        }

        std::string rawQ = parts[3];
        std::string processedQ;
        for (size_t i = 0; i < rawQ.size(); ++i) {
            if (rawQ[i] == '\\' && i + 1 < rawQ.size() && rawQ[i + 1] == 'n') {
                processedQ.push_back('\n');
                ++i;
            } else {
                processedQ.push_back(rawQ[i]);
            }
        }
        q.question = processedQ;
        outQuestions.push_back(std::move(q));
    }

    if (outQuestions.empty()) {
        if (outError) *outError = "No valid unseen questions parsed from: " + filePath;
        return false;
    }
    return true;
}

bool DatasetPipeline::verifyDisjointUnseenQuestions(
    const std::string& trainValDir,
    const std::vector<UnseenTestQuestion>& unseenQuestions,
    std::vector<std::string>* outOverlaps
) {
    namespace fs = std::filesystem;
    if (outOverlaps) outOverlaps->clear();
    if (!fs::exists(trainValDir) || !fs::is_directory(trainValDir)) {
        return false;
    }

    std::vector<std::pair<std::string, std::string>> docs;
    for (const auto& entry : fs::directory_iterator(trainValDir)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension();
            if (ext == ".txt" || ext == ".jsonl") {
                std::ifstream ifs(entry.path(), std::ios::binary);
                if (ifs) {
                    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                    docs.emplace_back(entry.path().string(), std::move(s));
                }
            }
        }
    }

    auto matches = checkContamination(docs, unseenQuestions, 0.72f);
    if (outOverlaps) {
        for (const auto& m : matches) {
            outOverlaps->push_back(m.unseenQuestion + " [" + m.matchType + " sim=" + std::to_string(m.similarity) + "]");
        }
    }
    return matches.empty();
}

} // namespace nexa
