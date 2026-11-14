/**
 * NEXA Native Training Engine - Trainer Pipeline Header
 * Manages dataset sequence batching, next-token prediction training loop,
 * AdamW optimization steps, loss reduction tracking, perplexity evaluation,
 * and checkpoint persistence tests.
 */

#ifndef NEXA_TRAINER_HPP
#define NEXA_TRAINER_HPP

#include "nano_model.hpp"
#include "optimizer.hpp"
#include "tokenizer.hpp"
#include "dataset.hpp"
#include <vector>
#include <string>

namespace nexa {

struct TrainingResult {
    bool success = false;
    uint64_t totalSteps = 0;
    float initialLoss = 0.0f;
    float finalLoss = 0.0f;
    float lossReductionPercent = 0.0f;
    float initialWeightNorm = 0.0f;
    float finalWeightNorm = 0.0f;
    float weightDeltaNorm = 0.0f;
    bool weightsUpdated = false;
    bool checkpointVerified = false;
    bool reloadedOutputIdentical = false;
    std::string checkpointPath;
    std::vector<float> lossHistory;
    int64_t totalTokensTrained = 0;
    double durationSeconds = 0.0;
};

struct EvaluationResult {
    float validationLoss = 0.0f;
    float perplexity = 0.0f;
    uint32_t sequencesEvaluated = 0;
    uint32_t tokensEvaluated = 0;
    double durationMs = 0.0;
};

struct EvalPoint {
    uint32_t step = 0;
    float lr = 0.0f;
    float trainLoss = 0.0f;
    float valLoss = 0.0f;
    float valPerplexity = 0.0f;
    float gradNorm = 0.0f;
};

struct PipelineTrainingResult {
    bool success = false;
    uint64_t totalSteps = 0;
    float initialTrainLoss = 0.0f;
    float finalTrainLoss = 0.0f;
    float initialValLoss = 0.0f;
    float finalValLoss = 0.0f;
    float bestValLoss = 0.0f;
    float initialPerplexity = 0.0f;
    float finalPerplexity = 0.0f;
    float bestPerplexity = 0.0f;
    uint32_t bestCheckpointStep = 0;
    bool overfittingDetected = false;
    float initialWeightNorm = 0.0f;
    float finalWeightNorm = 0.0f;
    float weightDeltaNorm = 0.0f;
    bool weightsUpdated = false;
    std::string checkpointPath;
    std::string bestCheckpointPath;
    double durationSeconds = 0.0;
    double tokensPerSecond = 0.0;
    double peakRssMb = 0.0;
    int64_t totalTokensTrained = 0;
    std::vector<EvalPoint> evalHistory;
    std::string sampleGeneration;
    std::string errorMessage;
};

struct TrainingPipelineConfig {
    uint32_t numSteps = 50;
    uint32_t epochs = 1;
    uint32_t batchSize = 1;
    uint32_t gradAccumSteps = 1;
    float learningRate = 3e-3f;
    float minLr = 1e-4f;
    uint32_t warmupSteps = 10;
    float weightDecay = 0.01f;
    float maxGradNorm = 1.0f;
    uint32_t evalInterval = 10;
    uint32_t earlyStopPatience = 0;
    uint64_t seed = 42;
    bool shuffleBatches = true;
    int32_t ignoreIndex = DatasetPipeline::IGNORE_INDEX;
    std::string checkpointPath = "data/checkpoints/nexa_nano_trained.nexa";
    std::string bestCheckpointPath = "";
    std::string resumeFromCheckpoint = "";
    std::string promptForSample = "NEXA";
};

struct CapacityBenchmarkEntry {
    std::string name;
    NanoConfig config;
    int64_t totalParameters = 0;
    ModelMemoryEstimate memoryEstimate;
    double avgStepMs = 0.0;
    double tokensPerSec = 0.0;
    double peakRssMb = 0.0;
    bool feasibleFor4GbCpu = true;
};

struct UnseenQuestionResult {
    std::string category;
    std::string expectedBehavior;
    std::string question;
    std::string generatedAnswer;
    uint32_t tokensGenerated = 0;
    double durationMs = 0.0;
    double tokensPerSecond = 0.0;
    float trigramRepetitionRate = 0.0f;
    float wordRepetitionRate = 0.0f;
    float relevanceScore = 0.0f;
    float coherenceScore = 0.0f;
    float factualAccuracyScore = 0.0f;
    float instructionFollowingScore = 0.0f;
    uint32_t keywordHits = 0;
    uint32_t totalKeywords = 0;
    bool cleanStop = false;
    bool nonEmptyCoherentFormat = false;
    bool relevanceHit = false;
};

struct UnseenEvalSummary {
    bool success = false;
    bool disjointVerified = false;
    uint32_t totalQuestions = 0;
    uint32_t nonEmptyResponses = 0;
    uint32_t cleanStopCount = 0;
    uint32_t relevanceHitCount = 0;
    uint32_t uncertaintyQuestionCount = 0;
    uint32_t uncertaintyMatchedCount = 0;
    double avgLatencyMs = 0.0;
    double avgTokensPerSec = 0.0;
    float avgRepetitionRate = 0.0f;
    float avgWordRepetitionRate = 0.0f;
    float avgRelevanceScore = 0.0f;
    float avgCoherenceScore = 0.0f;
    float avgFactualAccuracyScore = 0.0f;
    float avgInstructionFollowingScore = 0.0f;
    double peakRssMb = 0.0;
    std::vector<UnseenQuestionResult> results;
    std::string errorMessage;
};

class NexaTrainer {
public:
    static double getPeakRssMb();

    static TrainingResult trainOnSyntheticCorpus(
        NexaNanoModel& model,
        const std::string& textCorpus,
        uint32_t numSteps = 40,
        float learningRate = 3e-3f,
        uint32_t seqLen = 16,
        const std::string& checkpointDir = "data/checkpoints"
    );

    static EvaluationResult evaluateModel(
        NexaNanoModel& model,
        const std::vector<SequenceSample>& valSequences,
        int32_t ignoreIndex = DatasetPipeline::IGNORE_INDEX
    );

    static PipelineTrainingResult trainWithDataset(
        NexaNanoModel& model,
        const DatasetPipeline& dataset,
        uint32_t numSteps = 50,
        float learningRate = 3e-3f,
        const std::string& checkpointPath = "data/checkpoints/nexa_nano_trained.nexa"
    );

    static PipelineTrainingResult trainWithDatasetAdvanced(
        NexaNanoModel& model,
        const DatasetPipeline& dataset,
        const TrainingPipelineConfig& config,
        Tokenizer* tokenizer = nullptr
    );

    static std::vector<CapacityBenchmarkEntry> benchmarkConfigurations(uint32_t vocabSize = 768);

    static UnseenEvalSummary evaluateUnseenQuestions(
        NexaNanoModel& model,
        Tokenizer& tokenizer,
        const std::string& unseenQuestionsPath,
        const std::string& trainValDir,
        uint32_t maxNewTokens = 48,
        float temperature = 0.4f,
        uint64_t seed = 42
    );

    static bool verifyNumericalGradients(float* outMaxAbsDiff = nullptr);
    static bool verifyCheckpointRecovery(float* outMaxWeightDiff = nullptr);

    static int runTrainingVerificationTests();
    static int runGenerationAndEvaluationTests();
    static int runPhase34Tests();
    static int runPhase35Tests();
    static int runPhase36Tests();
};

} // namespace nexa

#endif // NEXA_TRAINER_HPP
