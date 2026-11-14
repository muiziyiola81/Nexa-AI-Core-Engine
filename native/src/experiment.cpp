/**
 * NEXA Native Training Engine - Experiment Manager Implementation
 */

#include "../include/nexa/experiment.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace nexa {

namespace {

std::string jsonEscapeExp(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (static_cast<unsigned char>(c) <= 0x1f) {
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        } else {
            o << c;
        }
    }
    return o.str();
}

} // namespace

std::string ExperimentManager::currentIsoTimestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tmUtc{};
    gmtime_r(&now, &tmUtc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmUtc);
    return std::string(buf);
}

std::string ExperimentManager::computeFileHashHex(const std::string& filePath) {
    std::ifstream ifs(filePath, std::ios::binary);
    if (!ifs.is_open()) return "missing";
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return DatasetPipeline::computeContentHashHex(content);
}

std::string ExperimentManager::serializeRecordJson(const ExperimentRecord& rec) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(6);
    o << "{\n";
    o << "  \"experimentId\": \"" << jsonEscapeExp(rec.experimentId) << "\",\n";
    o << "  \"timestampIso\": \"" << jsonEscapeExp(rec.timestampIso) << "\",\n";
    o << "  \"datasetLabel\": \"" << jsonEscapeExp(rec.datasetLabel) << "\",\n";
    o << "  \"modelConfig\": {\n";
    o << "    \"vocabSize\": " << rec.modelConfig.vocabSize << ",\n";
    o << "    \"dim\": " << rec.modelConfig.dim << ",\n";
    o << "    \"intermediateDim\": " << rec.modelConfig.intermediateDim << ",\n";
    o << "    \"nLayers\": " << rec.modelConfig.nLayers << ",\n";
    o << "    \"nHeads\": " << rec.modelConfig.nHeads << ",\n";
    o << "    \"maxSeqLen\": " << rec.modelConfig.maxSeqLen << ",\n";
    o << "    \"totalParameters\": " << rec.totalParameters << ",\n";
    o << "    \"architecturePrep\": {\n";
    o << "      \"rmsNormActive\": " << (rec.archPrep.rmsNormActive ? "true" : "false") << ",\n";
    o << "      \"ropePrepared\": " << (rec.archPrep.ropePrepared ? "true" : "false") << ",\n";
    o << "      \"ropeActiveInTraining\": " << (rec.archPrep.ropeActiveInTraining ? "true" : "false") << ",\n";
    o << "      \"tiedEmbeddingsPrepared\": " << (rec.archPrep.tiedEmbeddingsPrepared ? "true" : "false") << ",\n";
    o << "      \"tiedEmbeddingsActiveInTraining\": " << (rec.archPrep.tiedEmbeddingsActiveInTraining ? "true" : "false") << ",\n";
    o << "      \"dropoutActiveInTraining\": " << (rec.archPrep.dropoutActiveInTraining ? "true" : "false") << ",\n";
    o << "      \"checkpointFormatVersion\": " << rec.archPrep.checkpointFormatVersion << "\n";
    o << "    }\n";
    o << "  },\n";
    o << "  \"datasetInfo\": {\n";
    o << "    \"datasetVersion\": \"" << jsonEscapeExp(rec.datasetMeta.datasetVersion) << "\",\n";
    o << "    \"contentHashHex\": \"" << jsonEscapeExp(rec.datasetMeta.contentHashHex) << "\",\n";
    o << "    \"source\": \"" << jsonEscapeExp(rec.datasetMeta.source) << "\",\n";
    o << "    \"license\": \"" << jsonEscapeExp(rec.datasetMeta.license) << "\",\n";
    o << "    \"licenseVerified\": " << (rec.datasetMeta.licenseVerified ? "true" : "false") << ",\n";
    o << "    \"documentsCount\": " << rec.datasetMeta.documentsCount << ",\n";
    o << "    \"exampleCount\": " << rec.datasetMeta.exampleCount << ",\n";
    o << "    \"fileSizeBytes\": " << rec.datasetMeta.fileSizeBytes << ",\n";
    o << "    \"totalTokens\": " << rec.datasetMeta.totalTokens << ",\n";
    o << "    \"uniqueTokensCount\": " << rec.datasetMeta.uniqueTokensCount << ",\n";
    o << "    \"trainTokensCount\": " << rec.datasetMeta.trainTokensCount << ",\n";
    o << "    \"valTokensCount\": " << rec.datasetMeta.valTokensCount << ",\n";
    o << "    \"maskedTokensCount\": " << rec.datasetMeta.maskedTokensCount << ",\n";
    o << "    \"unmaskedTargetTokensCount\": " << rec.datasetMeta.unmaskedTargetTokensCount << ",\n";
    o << "    \"exactDuplicatesRemoved\": " << rec.datasetMeta.exactDuplicatesRemoved << ",\n";
    o << "    \"nearDuplicatesRemoved\": " << rec.datasetMeta.nearDuplicatesRemoved << ",\n";
    o << "    \"qualityIssuesRemoved\": " << rec.datasetMeta.qualityIssuesRemoved << ",\n";
    o << "    \"contaminationCount\": " << rec.datasetMeta.contaminationCount << ",\n";
    o << "    \"splitSeed\": " << rec.datasetMeta.splitSeed << "\n";
    o << "  },\n";
    o << "  \"tokenizerInfo\": {\n";
    o << "    \"type\": \"" << jsonEscapeExp(rec.tokenizerInfo.type) << "\",\n";
    o << "    \"path\": \"" << jsonEscapeExp(rec.tokenizerInfo.path) << "\",\n";
    o << "    \"versionHash\": \"" << jsonEscapeExp(rec.tokenizerInfo.versionHash) << "\",\n";
    o << "    \"vocabSize\": " << rec.tokenizerInfo.vocabSize << ",\n";
    o << "    \"numMerges\": " << rec.tokenizerInfo.numMerges << "\n";
    o << "  },\n";
    o << "  \"trainingSettings\": {\n";
    o << "    \"numSteps\": " << rec.trainingConfig.numSteps << ",\n";
    o << "    \"batchSize\": " << rec.trainingConfig.batchSize << ",\n";
    o << "    \"gradAccumSteps\": " << rec.trainingConfig.gradAccumSteps << ",\n";
    o << "    \"effectiveBatchSize\": " << (rec.trainingConfig.batchSize * rec.trainingConfig.gradAccumSteps) << ",\n";
    o << "    \"learningRate\": " << rec.trainingConfig.learningRate << ",\n";
    o << "    \"minLr\": " << rec.trainingConfig.minLr << ",\n";
    o << "    \"warmupSteps\": " << rec.trainingConfig.warmupSteps << ",\n";
    o << "    \"weightDecay\": " << rec.trainingConfig.weightDecay << ",\n";
    o << "    \"maxGradNorm\": " << rec.trainingConfig.maxGradNorm << ",\n";
    o << "    \"earlyStopPatience\": " << rec.trainingConfig.earlyStopPatience << ",\n";
    o << "    \"seed\": " << rec.trainingConfig.seed << ",\n";
    o << "    \"shuffleBatches\": " << (rec.trainingConfig.shuffleBatches ? "true" : "false") << ",\n";
    o << "    \"promptMasking\": " << (rec.trainingConfig.ignoreIndex == DatasetPipeline::IGNORE_INDEX ? "true" : "false") << "\n";
    o << "  },\n";
    o << "  \"results\": {\n";
    o << "    \"initialTrainLoss\": " << rec.trainingResult.initialTrainLoss << ",\n";
    o << "    \"finalTrainLoss\": " << rec.trainingResult.finalTrainLoss << ",\n";
    o << "    \"initialValLoss\": " << rec.trainingResult.initialValLoss << ",\n";
    o << "    \"finalValLoss\": " << rec.trainingResult.finalValLoss << ",\n";
    o << "    \"bestValLoss\": " << rec.trainingResult.bestValLoss << ",\n";
    o << "    \"bestPerplexity\": " << rec.trainingResult.bestPerplexity << ",\n";
    o << "    \"bestCheckpointStep\": " << rec.trainingResult.bestCheckpointStep << ",\n";
    o << "    \"overfittingDetected\": " << (rec.trainingResult.overfittingDetected ? "true" : "false") << ",\n";
    o << "    \"durationSeconds\": " << rec.trainingResult.durationSeconds << ",\n";
    o << "    \"tokensPerSecond\": " << rec.trainingResult.tokensPerSecond << ",\n";
    o << "    \"peakRssMb\": " << rec.trainingResult.peakRssMb << ",\n";
    o << "    \"bestCheckpointPath\": \"" << jsonEscapeExp(rec.trainingResult.bestCheckpointPath) << "\",\n";
    o << "    \"finalCheckpointPath\": \"" << jsonEscapeExp(rec.trainingResult.checkpointPath) << "\"\n";
    o << "  },\n";
    o << "  \"evalSummary\": {\n";
    o << "    \"evaluated\": " << (rec.evalSummary.evaluated ? "true" : "false") << ",\n";
    o << "    \"unseenSuitePath\": \"" << jsonEscapeExp(rec.evalSummary.unseenSuitePath) << "\",\n";
    o << "    \"disjointVerified\": " << (rec.evalSummary.disjointVerified ? "true" : "false") << ",\n";
    o << "    \"totalQuestions\": " << rec.evalSummary.totalQuestions << ",\n";
    o << "    \"relevanceHits\": " << rec.evalSummary.relevanceHits << ",\n";
    o << "    \"avgCoherenceScore\": " << rec.evalSummary.avgCoherenceScore << ",\n";
    o << "    \"avgFactualScore\": " << rec.evalSummary.avgFactualScore << ",\n";
    o << "    \"avgInstructionScore\": " << rec.evalSummary.avgInstructionScore << ",\n";
    o << "    \"avgRepetitionRate\": " << rec.evalSummary.avgRepetitionRate << ",\n";
    o << "    \"cleanStopCount\": " << rec.evalSummary.cleanStopCount << ",\n";
    o << "    \"uncertaintyMatched\": " << rec.evalSummary.uncertaintyMatched << ",\n";
    o << "    \"uncertaintyQuestions\": " << rec.evalSummary.uncertaintyQuestions << ",\n";
    o << "    \"avgLatencyMs\": " << rec.evalSummary.avgLatencyMs << "\n";
    o << "  }\n";
    o << "}\n";
    return o.str();
}

bool ExperimentManager::saveExperimentRecord(
    const ExperimentRecord& rec,
    const std::string& outDir,
    std::string* outFilePath
) {
    namespace fs = std::filesystem;
    fs::create_directories(outDir);
    std::string filePath = (fs::path(outDir) / (rec.experimentId + ".json")).string();
    std::string jsonStr = serializeRecordJson(rec);

    std::ofstream ofs(filePath, std::ios::binary);
    if (!ofs.is_open()) return false;
    ofs.write(jsonStr.data(), static_cast<std::streamsize>(jsonStr.size()));
    ofs.close();

    // Also append compact single-line summary to registry.jsonl
    std::string compact = jsonStr;
    for (char& ch : compact) {
        if (ch == '\n' || ch == '\r') ch = ' ';
    }
    std::string regPath = (fs::path(outDir) / "experiment_registry.jsonl").string();
    std::ofstream regOfs(regPath, std::ios::app | std::ios::binary);
    if (regOfs.is_open()) {
        regOfs << compact << "\n";
        regOfs.close();
    }

    if (outFilePath) *outFilePath = filePath;
    return true;
}

} // namespace nexa
