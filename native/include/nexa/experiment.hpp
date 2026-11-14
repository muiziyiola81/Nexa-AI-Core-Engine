/**
 * NEXA Native Training Engine - Experiment Manager Header
 * Records reproducible experiment configurations, dataset versions, provenance,
 * tokenizer versions, training hyperparameters, loss/perplexity trajectories,
 * and multi-dimensional unseen evaluation results.
 */

#ifndef NEXA_EXPERIMENT_HPP
#define NEXA_EXPERIMENT_HPP

#include "nano_model.hpp"
#include "dataset.hpp"
#include "trainer.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace nexa {

struct TokenizerVersionInfo {
    std::string type = "bpe";
    std::string path;
    std::string versionHash;
    uint32_t vocabSize = 0;
    uint32_t numMerges = 0;
};

struct ExperimentEvalSummary {
    bool evaluated = false;
    std::string unseenSuitePath;
    bool disjointVerified = false;
    uint32_t totalQuestions = 0;
    uint32_t relevanceHits = 0;
    double avgCoherenceScore = 0.0;
    double avgFactualScore = 0.0;
    double avgInstructionScore = 0.0;
    float avgRepetitionRate = 0.0f;
    uint32_t cleanStopCount = 0;
    uint32_t uncertaintyMatched = 0;
    uint32_t uncertaintyQuestions = 0;
    double avgLatencyMs = 0.0;
};

struct ExperimentRecord {
    std::string experimentId;
    std::string timestampIso;
    std::string datasetLabel;
    NanoConfig modelConfig;
    int64_t totalParameters = 0;
    ArchitecturePrepSpec archPrep;
    DatasetMetadata datasetMeta;
    DatasetQualityReport qualityReport;
    TokenizerVersionInfo tokenizerInfo;
    TrainingPipelineConfig trainingConfig;
    PipelineTrainingResult trainingResult;
    ExperimentEvalSummary evalSummary;
};

class ExperimentManager {
public:
    static std::string currentIsoTimestamp();
    static std::string computeFileHashHex(const std::string& filePath);
    static std::string serializeRecordJson(const ExperimentRecord& rec);

    static bool saveExperimentRecord(
        const ExperimentRecord& rec,
        const std::string& outDir = "data/experiments",
        std::string* outFilePath = nullptr
    );
};

} // namespace nexa

#endif // NEXA_EXPERIMENT_HPP
