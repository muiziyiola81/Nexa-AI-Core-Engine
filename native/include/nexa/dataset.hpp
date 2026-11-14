/**
 * NEXA Native Training Engine - Dataset Pipeline Header
 * Implements plain-text dataset loading, multi-file ingestion, deduplication,
 * strict UTF-8 validation, tokenization, train/val sequence splitting, and metadata tracking.
 */

#ifndef NEXA_DATASET_HPP
#define NEXA_DATASET_HPP

#include "tokenizer.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace nexa {

struct DatasetSourceRecord {
    std::string id;
    std::string path;
    std::string format = "unknown"; // "text", "instruction_txt", "instruction_jsonl"
    std::string domain = "general";
    std::string source = "unspecified";
    std::string license = "UNVERIFIED";
    bool licenseVerified = false;
    uint64_t bytes = 0;
    uint32_t documents = 0;
    uint32_t examples = 0;
};

struct QualityIssueRecord {
    size_t exampleIndex = 0;
    std::string sourceFile;
    std::string issueType;
    std::string detail;
};

struct DatasetQualityReport {
    uint32_t totalInputBlocks = 0;
    uint32_t acceptedBlocks = 0;
    uint32_t rejectedBlocks = 0;
    uint32_t utf8ErrorCount = 0;
    uint32_t corruptedTextCount = 0;
    uint32_t incompleteSentenceCount = 0;
    uint32_t repetitiveExampleCount = 0;
    uint32_t contradictoryAnswerCount = 0;
    uint32_t lowInformationCount = 0;
    uint32_t formattingErrorCount = 0;
    uint32_t malformedConversationCount = 0;
    uint32_t exactDuplicatesRemoved = 0;
    uint32_t nearDuplicatesRemoved = 0;
    uint32_t contaminationCount = 0;
    float duplicationRate = 0.0f;
    bool passedQualityGate = false;
    std::vector<QualityIssueRecord> issues;
};

struct ContaminationMatch {
    std::string unseenQuestion;
    std::string matchedSource;
    std::string matchType; // "exact_substring", "normalized_match", "ngram_jaccard"
    float similarity = 0.0f;
};

struct DatasetMetadata {
    std::string filePath;
    std::vector<std::string> sourceFiles;
    std::vector<DatasetSourceRecord> sourceRecords;
    std::string source = "unspecified";
    std::string license = "UNVERIFIED";
    bool licenseVerified = false;
    std::string datasetVersion = "unversioned";
    std::string contentHashHex;
    uint32_t documentsCount = 0;
    uint64_t fileSizeBytes = 0;
    uint64_t totalCharacters = 0;
    uint64_t totalTokens = 0;
    uint64_t uniqueTokensCount = 0;
    uint64_t trainTokensCount = 0;
    uint64_t valTokensCount = 0;
    uint64_t maskedTokensCount = 0;
    uint64_t unmaskedTargetTokensCount = 0;
    uint32_t trainSequencesCount = 0;
    uint32_t valSequencesCount = 0;
    uint32_t uniqueLines = 0;
    uint32_t duplicatesRemoved = 0;
    uint32_t exactDuplicatesRemoved = 0;
    uint32_t nearDuplicatesRemoved = 0;
    uint32_t invalidLinesRemoved = 0;
    uint32_t qualityIssuesRemoved = 0;
    uint32_t contaminationCount = 0;
    uint32_t exampleCount = 0;
    uint32_t trainExamplesCount = 0;
    uint32_t valExamplesCount = 0;
    float avgTokensPerExample = 0.0f;
    float splitRatio = 0.8f;
    uint64_t splitSeed = 42;
    uint32_t seqLen = 32;
    bool isValid = false;
    std::string errorMessage;
};

struct SequenceSample {
    std::vector<int32_t> inputs;
    std::vector<int32_t> targets;
};

struct UnseenTestQuestion {
    std::string category;
    std::string expectedBehavior; // "factual", "instruction", "conversation", "followup", "uncertainty"
    std::vector<std::string> keywords;
    std::string question;
};

struct RawExampleBlock {
    std::string text;
    std::string sourceFile;
    std::string source = "unspecified";
    std::string license = "UNVERIFIED";
    bool isInstructionFormat = true;
};

class DatasetPipeline {
private:
    DatasetMetadata metadata_;
    DatasetQualityReport qualityReport_;
    std::vector<int32_t> allTokens_;
    std::vector<int32_t> trainTokens_;
    std::vector<int32_t> valTokens_;
    std::vector<SequenceSample> trainSequences_;
    std::vector<SequenceSample> valSequences_;
    std::vector<std::string> acceptedExampleTexts_;

public:
    static constexpr int32_t IGNORE_INDEX = -100;

    DatasetPipeline() = default;

    /**
     * Strict RFC 3629 UTF-8 validation.
     * Returns true if valid UTF-8. If false, outErrorOffset contains byte index.
     */
    static bool validateUtf8(const std::string& data, size_t* outErrorOffset = nullptr);

    /**
     * Deterministic FNV-1a 64-bit hex hash for dataset/split versioning.
     */
    static std::string computeContentHashHex(const std::string& data);

    /**
     * Checks whether a declared license string is an explicitly known permissive license.
     * Unknown, empty, "UNKNOWN", or "UNVERIFIED" strings always return false and are
     * never silently labeled MIT.
     */
    static bool isVerifiedPermissiveLicense(const std::string& license);

    /**
     * Parses a structured JSONL instruction/conversation line into canonical "User:/Assistant:" block text.
     */
    static bool parseJsonlInstructionLine(
        const std::string& jsonLine,
        std::string& outBlockText,
        std::string& outSource,
        std::string& outLicense,
        std::string* outError = nullptr
    );

    /**
     * Runs comprehensive quality checks (corrupted text, incomplete sentences, repetition,
     * contradictions, low-information, formatting, conversation alternation, exact & near-duplicates).
     */
    static DatasetQualityReport analyzeAndFilterBlocks(
        const std::vector<RawExampleBlock>& inputBlocks,
        std::vector<RawExampleBlock>& outAcceptedBlocks,
        float nearDuplicateThreshold = 0.85f,
        bool filterFailed = true
    );

    /**
     * Checks for both exact substring leakage and normalized / n-gram Jaccard contamination
     * between held-out unseen evaluation questions and training/validation texts.
     */
    static std::vector<ContaminationMatch> checkContamination(
        const std::vector<std::pair<std::string, std::string>>& sourceDocs,
        const std::vector<UnseenTestQuestion>& unseenQuestions,
        float jaccardThreshold = 0.72f
    );

    /**
     * Loads a single text file, validates UTF-8, tokenizes using tokenizer,
     * and splits into train and validation sequence sets.
     */
    bool loadFromFile(
        const std::string& filePath,
        Tokenizer& tokenizer,
        float trainValSplit = 0.8f,
        uint32_t seqLen = 32
    );

    /**
     * Loads multiple text files, performs deduplication and strict UTF-8 validation,
     * tokenizes, and builds train/validation sequences.
     */
    bool loadFromFiles(
        const std::vector<std::string>& filePaths,
        Tokenizer& tokenizer,
        float trainValSplit = 0.8f,
        uint32_t seqLen = 32,
        bool deduplicate = true
    );

    /**
     * Loads all .txt files from a directory.
     */
    bool loadFromDirectory(
        const std::string& dirPath,
        Tokenizer& tokenizer,
        float trainValSplit = 0.8f,
        uint32_t seqLen = 32,
        bool deduplicate = true
    );

    /**
     * Loads instruction/QA blocks separated by blank lines from multiple files (.txt or .jsonl),
     * deduplicates, shuffles at the example level before splitting into train/val,
     * and optionally masks prompt tokens (User: / Assistant: prefix) with IGNORE_INDEX.
     */
    bool loadInstructionExamplesFromFiles(
        const std::vector<std::string>& filePaths,
        Tokenizer& tokenizer,
        float trainValSplit = 0.85f,
        uint32_t seqLen = 64,
        bool maskPrompts = true,
        uint64_t splitSeed = 42,
        const std::string& source = "unspecified",
        const std::string& license = "UNVERIFIED",
        bool enforceQualityChecks = false,
        const std::string& unseenTestPath = ""
    );

    bool loadInstructionExamplesFromDirectory(
        const std::string& dirPath,
        Tokenizer& tokenizer,
        float trainValSplit = 0.85f,
        uint32_t seqLen = 64,
        bool maskPrompts = true,
        uint64_t splitSeed = 42,
        const std::string& source = "unspecified",
        const std::string& license = "UNVERIFIED",
        bool enforceQualityChecks = false,
        const std::string& unseenTestPath = ""
    );

    /**
     * Loads a managed, versioned dataset directory or manifest JSON, validates provenance/licensing,
     * ingests both .txt and .jsonl instruction/text sources, applies quality filtering & near-duplicate
     * pruning, checks contamination against unseen evaluation questions, and builds reproducible splits.
     */
    bool loadManagedDataset(
        const std::string& datasetDirOrManifest,
        Tokenizer& tokenizer,
        float trainValSplit = 0.85f,
        uint32_t seqLen = 64,
        bool maskPrompts = true,
        uint64_t splitSeed = 42,
        const std::string& unseenTestPath = "",
        bool requireVerifiedLicense = false,
        bool rejectIfContaminated = true
    );

    /**
     * Loads unseen evaluation test questions (CATEGORY|BEHAVIOR|KEYWORDS|QUESTION).
     */
    static bool loadUnseenQuestions(
        const std::string& filePath,
        std::vector<UnseenTestQuestion>& outQuestions,
        std::string* outError = nullptr
    );

    /**
     * Verifies that none of the unseen test questions appear verbatim or via high n-gram overlap
     * inside any training file.
     */
    static bool verifyDisjointUnseenQuestions(
        const std::string& trainValDir,
        const std::vector<UnseenTestQuestion>& unseenQuestions,
        std::vector<std::string>* outOverlaps = nullptr
    );

    /**
     * Loads directly from an in-memory string.
     */
    bool loadFromString(
        const std::string& text,
        const std::string& datasetName,
        Tokenizer& tokenizer,
        float trainValSplit = 0.8f,
        uint32_t seqLen = 32
    );

    const DatasetMetadata& metadata() const { return metadata_; }
    const DatasetQualityReport& qualityReport() const { return qualityReport_; }
    const std::vector<int32_t>& allTokens() const { return allTokens_; }
    const std::vector<int32_t>& trainTokens() const { return trainTokens_; }
    const std::vector<int32_t>& valTokens() const { return valTokens_; }
    const std::vector<SequenceSample>& trainSequences() const { return trainSequences_; }
    const std::vector<SequenceSample>& valSequences() const { return valSequences_; }
    const std::vector<std::string>& acceptedExampleTexts() const { return acceptedExampleTexts_; }

private:
    void buildSequences(uint32_t seqLen);
};

} // namespace nexa

#endif // NEXA_DATASET_HPP
