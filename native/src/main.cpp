/**
 * NEXA Native Runtime - Standalone C++ Engine Executable
 * CLI interface for tensor benchmarks, model inspection, native execution,
 * NEXA-1 Nano training pipeline, and autoregressive text generation.
 */

#include "../include/nexa/tensor.hpp"
#include "../include/nexa/nn.hpp"
#include "../include/nexa/tokenizer.hpp"
#include "../include/nexa/model_loader.hpp"
#include "../include/nexa/autograd.hpp"
#include "../include/nexa/optimizer.hpp"
#include "../include/nexa/nano_model.hpp"
#include "../include/nexa/dataset.hpp"
#include "../include/nexa/trainer.hpp"
#include "../include/nexa/experiment.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <fstream>

using namespace nexa;

void printBanner() {
    std::cout << "================================================================" << std::endl;
    std::cout << "       NEXA NATIVE RUNTIME & TRAINING ENGINE (C++20 CPU)        " << std::endl;
    std::cout << "================================================================" << std::endl;
}

static std::string escapeJson(const std::string& s) {
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

static std::string trimEdges(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t' || s[start] == '\r' || s[start] == '\n')) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r' || s[end - 1] == '\n')) {
        --end;
    }
    return s.substr(start, end - start);
}

static std::string normalizeConversationPrompt(const std::string& rawInput) {
    std::string trimmed = trimEdges(rawInput);
    if (trimmed.empty()) return "User:\nAssistant:";

    bool hasCanonicalRole =
        trimmed.rfind("User:", 0) == 0 ||
        trimmed.rfind("System:", 0) == 0 ||
        trimmed.rfind("Assistant:", 0) == 0 ||
        trimmed.find("\nUser:") != std::string::npos ||
        trimmed.find("\nSystem:") != std::string::npos ||
        trimmed.find("\nAssistant:") != std::string::npos;

    if (!hasCanonicalRole) {
        return "User: " + trimmed + "\nAssistant:";
    }

    // Already formatted with canonical System:/User:/Assistant: roles.
    // Ensure it ends with Assistant: so the model generates the assistant turn.
    const std::string suffix = "Assistant:";
    if (trimmed.size() >= suffix.size() &&
        trimmed.compare(trimmed.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return trimmed;
    }
    return trimmed + "\nAssistant:";
}

static std::string cleanAssistantReply(const std::string& rawReply) {
    std::string out = rawReply;
    // Strip a single leading space emitted by word-with-leading-space BPE tokens after "Assistant:"
    if (!out.empty() && out.front() == ' ') {
        out.erase(0, 1);
    }
    while (!out.empty() && (out.front() == '\r' || out.front() == '\n')) {
        out.erase(0, 1);
    }
    while (!out.empty() && (out.back() == '\r' || out.back() == '\n' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

int runTensorTest() {
    std::cout << "\n[TEST 1] Matrix Multiplication Math Verification..." << std::endl;
    Tensor A(Shape({2, 3}));
    Tensor B(Shape({3, 2}));
    Tensor C(Shape({2, 2}));

    float* aPtr = A.dataFloat();
    aPtr[0] = 1.0f; aPtr[1] = 2.0f; aPtr[2] = 3.0f;
    aPtr[3] = 4.0f; aPtr[4] = 5.0f; aPtr[5] = 6.0f;

    float* bPtr = B.dataFloat();
    bPtr[0] = 7.0f; bPtr[1] = 8.0f;
    bPtr[2] = 9.0f; bPtr[3] = 1.0f;
    bPtr[4] = 2.0f; bPtr[5] = 3.0f;

    Tensor::matmul(A, B, C);
    const float* cPtr = C.dataFloat();

    std::cout << "C = [[" << cPtr[0] << ", " << cPtr[1] << "], [" << cPtr[2] << ", " << cPtr[3] << "]]" << std::endl;

    if (std::abs(cPtr[0] - 31.0f) > 1e-4f || std::abs(cPtr[1] - 19.0f) > 1e-4f ||
        std::abs(cPtr[2] - 85.0f) > 1e-4f || std::abs(cPtr[3] - 55.0f) > 1e-4f) {
        std::cerr << "FAIL: Matmul calculation mismatch!" << std::endl;
        return 1;
    }
    std::cout << "Matmul assertion passed (ground truth matches)." << std::endl;

    std::cout << "\n[TEST 2] Numerically Stable Softmax Verification..." << std::endl;
    Tensor sIn(Shape({1, 4}));
    Tensor sOut(Shape({1, 4}));
    float* sInPtr = sIn.dataFloat();
    sInPtr[0] = 1000.0f;
    sInPtr[1] = 1001.0f;
    sInPtr[2] = 1002.0f;
    sInPtr[3] = 1000.0f;

    Tensor::softmax(sIn, sOut);
    const float* sOutPtr = sOut.dataFloat();
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += sOutPtr[i];
    }
    std::cout << "Softmax sum = " << sum << " (expected 1.0)" << std::endl;
    if (std::abs(sum - 1.0f) > 1e-5f) {
        std::cerr << "FAIL: Softmax does not sum to 1.0!" << std::endl;
        return 1;
    }
    std::cout << "Softmax assertion passed." << std::endl;

    std::cout << "\n[TEST 3] RMSNorm Invariant Verification..." << std::endl;
    Tensor rIn(Shape({1, 64}));
    Tensor rWeight(Shape({64}));
    Tensor rOut(Shape({1, 64}));
    rIn.fillConstant(2.5f);
    rWeight.fillConstant(1.0f);

    RMSNorm::forward(rIn, rWeight, rOut, 1e-5f);
    const float* rOutPtr = rOut.dataFloat();
    std::cout << "RMSNorm of constant 2.5 with weight 1.0 = " << rOutPtr[0] << " (expected ~1.0)" << std::endl;
    if (std::abs(rOutPtr[0] - 1.0f) > 1e-3f) {
        std::cerr << "FAIL: RMSNorm invariant violated!" << std::endl;
        return 1;
    }
    std::cout << "RMSNorm assertion passed." << std::endl;

    std::cout << "\n[TEST 4] Tokenizer Byte Round-Trip..." << std::endl;
    NexaByteTokenizer tokenizer;
    std::string sample = "Hello NEXA native runtime!";
    auto tokens = tokenizer.encode(sample, true, true);
    std::cout << "Encoded tokens (" << tokens.size() << "): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;

    std::string decoded = tokenizer.decode(tokens, true);
    std::cout << "Decoded text: \"" << decoded << "\"" << std::endl;
    if (decoded != sample) {
        std::cerr << "FAIL: Decoded text did not match input!" << std::endl;
        return 1;
    }
    std::cout << "Tokenizer round-trip passed." << std::endl;

    std::cout << "\nALL 4 NATIVE TENSOR & NN UNIT TESTS PASSED SUCCESSFULLY." << std::endl;
    return 0;
}

int runTensorTestJson() {
    Tensor A(Shape({2, 3}));
    Tensor B(Shape({3, 2}));
    Tensor C(Shape({2, 2}));

    float* aPtr = A.dataFloat();
    aPtr[0] = 1.0f; aPtr[1] = 2.0f; aPtr[2] = 3.0f;
    aPtr[3] = 4.0f; aPtr[4] = 5.0f; aPtr[5] = 6.0f;

    float* bPtr = B.dataFloat();
    bPtr[0] = 7.0f; bPtr[1] = 8.0f;
    bPtr[2] = 9.0f; bPtr[3] = 1.0f;
    bPtr[4] = 2.0f; bPtr[5] = 3.0f;

    Tensor::matmul(A, B, C);
    const float* cPtr = C.dataFloat();

    bool matmulPassed = (std::abs(cPtr[0] - 31.0f) <= 1e-4f && std::abs(cPtr[1] - 19.0f) <= 1e-4f &&
                         std::abs(cPtr[2] - 85.0f) <= 1e-4f && std::abs(cPtr[3] - 55.0f) <= 1e-4f);

    Tensor sIn(Shape({1, 4}));
    Tensor sOut(Shape({1, 4}));
    float* sInPtr = sIn.dataFloat();
    sInPtr[0] = 1000.0f;
    sInPtr[1] = 1001.0f;
    sInPtr[2] = 1002.0f;
    sInPtr[3] = 1000.0f;

    Tensor::softmax(sIn, sOut);
    const float* sOutPtr = sOut.dataFloat();
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += sOutPtr[i];
    bool softmaxPassed = (std::abs(sum - 1.0f) <= 1e-5f);

    Tensor rIn(Shape({1, 64}));
    Tensor rWeight(Shape({64}));
    Tensor rOut(Shape({1, 64}));
    rIn.fillConstant(2.5f);
    rWeight.fillConstant(1.0f);

    RMSNorm::forward(rIn, rWeight, rOut, 1e-5f);
    const float* rOutPtr = rOut.dataFloat();
    bool rmsnormPassed = (std::abs(rOutPtr[0] - 1.0f) <= 1e-3f);

    NexaByteTokenizer tokenizer;
    std::string sample = "Hello NEXA native runtime!";
    auto tokens = tokenizer.encode(sample, true, true);
    std::string decoded = tokenizer.decode(tokens, true);
    bool tokenizerPassed = (decoded == sample);

    int passedCount = (matmulPassed ? 1 : 0) + (softmaxPassed ? 1 : 0) +
                      (rmsnormPassed ? 1 : 0) + (tokenizerPassed ? 1 : 0);

    std::cout << "{"
              << "\"status\":\"ok\","
              << "\"tests_passed\":" << passedCount << ","
              << "\"matmul\":{\"result\":[[" << cPtr[0] << "," << cPtr[1] << "],[" << cPtr[2] << "," << cPtr[3] << "]],\"passed\":" << (matmulPassed ? "true" : "false") << "},"
              << "\"softmax\":{\"sum\":" << sum << ",\"passed\":" << (softmaxPassed ? "true" : "false") << "},"
              << "\"rmsnorm\":{\"val\":" << rOutPtr[0] << ",\"passed\":" << (rmsnormPassed ? "true" : "false") << "},"
              << "\"tokenizer\":{\"tokens_count\":" << tokens.size() << ",\"passed\":" << (tokenizerPassed ? "true" : "false") << "}"
              << "}" << std::endl;

    return (passedCount == 4) ? 0 : 1;
}

int runForwardPassJson(uint32_t dim = 128, uint32_t seqLen = 8) {
    auto t0 = std::chrono::high_resolution_clock::now();

    TransformerHyperparams hp;
    hp.dim = dim;
    hp.nLayers = 1;
    hp.nHeads = 4;
    hp.nKvHeads = 4;
    hp.vocabSize = 256;
    hp.intermediateDim = dim * 2;
    hp.maxSeqLen = seqLen;
    hp.normEps = 1e-5f;
    hp.ropeTheta = 10000.0f;

    const int64_t sDim = static_cast<int64_t>(dim);
    const int64_t sSeq = static_cast<int64_t>(seqLen);
    const int64_t sInter = static_cast<int64_t>(hp.intermediateDim);

    Tensor input(Shape({sSeq, sDim}));
    input.fillConstant(0.1f);

    Tensor attnNorm(Shape({sDim}));
    attnNorm.fillConstant(1.0f);

    Tensor wQ(Shape({sDim, sDim}));
    Tensor wK(Shape({sDim, sDim}));
    Tensor wV(Shape({sDim, sDim}));
    Tensor wOut(Shape({sDim, sDim}));
    wQ.fillConstant(0.01f);
    wK.fillConstant(0.01f);
    wV.fillConstant(0.01f);
    wOut.fillConstant(0.01f);

    Tensor ffnNorm(Shape({sDim}));
    ffnNorm.fillConstant(1.0f);

    Tensor wGate(Shape({sDim, sInter}));
    Tensor wUp(Shape({sDim, sInter}));
    Tensor wDown(Shape({sInter, sDim}));
    wGate.fillConstant(0.01f);
    wUp.fillConstant(0.01f);
    wDown.fillConstant(0.01f);

    Tensor output(Shape({sSeq, sDim}));

    TransformerBlock::forward(input, attnNorm, wQ, wK, wV, wOut, ffnNorm, wGate, wUp, wDown, output, hp, 0);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    const float* outPtr = output.dataFloat();
    float checksum = 0.0f;
    for (int64_t i = 0; i < output.numel(); ++i) checksum += outPtr[i];

    std::cout << "{"
              << "\"status\":\"ok\","
              << "\"runtime\":\"C++20 Native Engine\","
              << "\"dim\":" << dim << ","
              << "\"seq_len\":" << seqLen << ","
              << "\"duration_us\":" << elapsedUs << ","
              << "\"output_checksum\":" << checksum << ","
              << "\"tensor_elements\":" << output.numel()
              << "}" << std::endl;

    return 0;
}

int runNanoTraining(uint32_t numSteps = 40, float lr = 3e-3f, bool jsonOutput = false) {
    NanoConfig cfg;
    cfg.vocabSize = 256;
    cfg.dim = 64;
    cfg.intermediateDim = 128;
    cfg.nLayers = 2;
    cfg.nHeads = 2;
    cfg.maxSeqLen = 32;

    NexaNanoModel model(cfg, 42); // seed 42

    std::string syntheticCorpus =
        "NEXA AI CORE ENGINE INDEPENDENT NATIVE ARCHITECTURE\n"
        "TRAINING NEXA-1 NANO FROM RANDOMLY INITIALIZED WEIGHTS\n"
        "ZERO THIRD-PARTY RUNTIME DEPENDENCIES\n"
        "SELF-HOSTED AUTOGRAD AND ADAMW OPTIMIZATION LOOP VERIFIED\n";

    TrainingResult res = NexaTrainer::trainOnSyntheticCorpus(model, syntheticCorpus, numSteps, lr, 16);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (res.success ? "ok" : "error") << "\","
                  << "\"engine\":\"NEXA Native Training Engine (C++20)\","
                  << "\"total_steps\":" << res.totalSteps << ","
                  << "\"initial_loss\":" << res.initialLoss << ","
                  << "\"final_loss\":" << res.finalLoss << ","
                  << "\"loss_reduction_pct\":" << res.lossReductionPercent << ","
                  << "\"initial_weight_norm\":" << res.initialWeightNorm << ","
                  << "\"final_weight_norm\":" << res.finalWeightNorm << ","
                  << "\"weight_delta_norm\":" << res.weightDeltaNorm << ","
                  << "\"weights_updated\":" << (res.weightsUpdated ? "true" : "false") << ","
                  << "\"checkpoint_verified\":" << (res.checkpointVerified ? "true" : "false") << ","
                  << "\"reloaded_output_identical\":" << (res.reloadedOutputIdentical ? "true" : "false") << ","
                  << "\"checkpoint_path\":\"" << res.checkpointPath << "\","
                  << "\"duration_s\":" << res.durationSeconds
                  << "}" << std::endl;
    }

    return res.success ? 0 : 1;
}

int runGeneration(
    const std::string& ckptPath,
    const std::string& prompt,
    uint32_t maxTokens = 32,
    float temp = 0.7f,
    int32_t topK = 40,
    float topP = 0.9f,
    float repPenalty = 1.15f,
    uint64_t seed = 42,
    bool jsonOutput = false,
    const std::string& nbtPath = ""
) {
    auto val = NexaNanoModel::validateCheckpoint(ckptPath);
    if (!val.valid) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(val.error) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Checkpoint validation failed: " << val.error << std::endl;
        }
        return 1;
    }

    NexaNanoModel model(val.config, 42);
    std::string loadErr;
    if (!model.loadCheckpoint(ckptPath, nullptr, nullptr, nullptr, &loadErr)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(loadErr) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Failed to load checkpoint weights: " << loadErr << std::endl;
        }
        return 1;
    }

    std::unique_ptr<Tokenizer> tok;
    if (!nbtPath.empty()) {
        if (!std::filesystem::exists(nbtPath)) {
            std::string err = "Tokenizer file not found: " + nbtPath;
            if (jsonOutput) {
                std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
            } else {
                std::cerr << "Error: " << err << std::endl;
            }
            return 1;
        }
        auto bpe = std::make_unique<NexaBpeTokenizer>();
        if (!bpe->load(nbtPath)) {
            std::string err = "Failed to load BPE tokenizer: " + nbtPath;
            if (jsonOutput) {
                std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
            } else {
                std::cerr << "Error: " << err << std::endl;
            }
            return 1;
        }
        tok = std::move(bpe);
    } else if (val.config.vocabSize <= 260) {
        tok = std::make_unique<NexaByteTokenizer>();
    } else {
        std::string err = "Checkpoint requires a BPE .nbt tokenizer (vocab_size=" + std::to_string(val.config.vocabSize) + ")";
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: " << err << std::endl;
        }
        return 1;
    }

    if (!nbtPath.empty() && tok->vocabSize() != val.config.vocabSize) {
        std::string vErr = "Tokenizer vocabulary size (" + std::to_string(tok->vocabSize()) +
                           ") does not match checkpoint vocabulary size (" + std::to_string(val.config.vocabSize) + ")";
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(vErr) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: " << vErr << std::endl;
        }
        return 1;
    }

    GenerationConfig genCfg;
    genCfg.maxNewTokens = maxTokens;
    genCfg.temperature = temp;
    genCfg.topK = topK;
    genCfg.topP = topP;
    genCfg.repetitionPenalty = repPenalty;
    genCfg.seed = seed;

    auto genRes = model.generateText(prompt, *tok, genCfg);
    if (!genRes.success) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(genRes.errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error during generation: " << genRes.errorMessage << std::endl;
        }
        return 1;
    }

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"ok\","
                  << "\"checkpoint\":\"" << escapeJson(ckptPath) << "\","
                  << "\"prompt\":\"" << escapeJson(prompt) << "\","
                  << "\"generated_text\":\"" << escapeJson(genRes.generatedText) << "\","
                  << "\"full_text\":\"" << escapeJson(genRes.fullText) << "\","
                  << "\"prompt_tokens\":" << genRes.promptTokens.size() << ","
                  << "\"tokens_generated\":" << genRes.tokensGenerated << ","
                  << "\"max_seq_len\":" << val.config.maxSeqLen << ","
                  << "\"vocab_size\":" << val.config.vocabSize << ","
                  << "\"use_rope\":" << (val.config.useRope ? "true" : "false") << ","
                  << "\"tie_embeddings\":" << (val.config.tieEmbeddings ? "true" : "false") << ","
                  << "\"duration_ms\":" << genRes.durationMs << ","
                  << "\"tokens_per_second\":" << genRes.tokensPerSecond << ","
                  << "\"finish_reason\":\"" << genRes.finishReason << "\""
                  << "}" << std::endl;
    } else {
        std::cout << "\n----------------------------------------------------------------" << std::endl;
        std::cout << "Loaded Checkpoint: " << ckptPath << std::endl;
        std::cout << "Architecture: " << val.config.nLayers << "L, " << val.config.nHeads << "H, " << val.config.dim << "D"
                  << " (RoPE=" << (val.config.useRope ? "on" : "off")
                  << ", Tied=" << (val.config.tieEmbeddings ? "on" : "off") << ")" << std::endl;
        std::cout << "Prompt:            \"" << prompt << "\"" << std::endl;
        std::cout << "Generated Output:  \"" << genRes.generatedText << "\"" << std::endl;
        std::cout << "Full Text:         \"" << genRes.fullText << "\"" << std::endl;
        std::cout << "Prompt Tokens:     " << genRes.promptTokens.size() << " / " << val.config.maxSeqLen << std::endl;
        std::cout << "Tokens Generated:  " << genRes.tokensGenerated << std::endl;
        std::cout << "Duration:          " << std::fixed << std::setprecision(2) << genRes.durationMs << " ms" << std::endl;
        std::cout << "Inference Speed:   " << std::fixed << std::setprecision(1) << genRes.tokensPerSecond << " tokens/sec" << std::endl;
        std::cout << "Finish Reason:     " << genRes.finishReason << std::endl;
        std::cout << "----------------------------------------------------------------\n" << std::endl;
    }

    return 0;
}

int runDatasetPipelineTraining(
    const std::string& datasetPath,
    uint32_t steps = 50,
    float lr = 3e-3f,
    uint32_t seqLen = 16,
    float split = 0.8f,
    const std::string& ckptPath = "data/checkpoints/nexa_nano_trained.nexa",
    bool jsonOutput = false
) {
    NexaByteTokenizer tokenizer;
    DatasetPipeline pipeline;
    if (!pipeline.loadFromFile(datasetPath, tokenizer, split, seqLen)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(pipeline.metadata().errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error loading dataset: " << pipeline.metadata().errorMessage << std::endl;
        }
        return 1;
    }

    NanoConfig cfg;
    cfg.vocabSize = 256;
    cfg.dim = 64;
    cfg.intermediateDim = 128;
    cfg.nLayers = 2;
    cfg.nHeads = 2;
    cfg.maxSeqLen = seqLen * 2;

    NexaNanoModel model(cfg, 1337);
    auto res = NexaTrainer::trainWithDataset(model, pipeline, steps, lr, ckptPath);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (res.success ? "ok" : "error") << "\","
                  << "\"total_steps\":" << res.totalSteps << ","
                  << "\"initial_train_loss\":" << res.initialTrainLoss << ","
                  << "\"final_train_loss\":" << res.finalTrainLoss << ","
                  << "\"initial_val_loss\":" << res.initialValLoss << ","
                  << "\"final_val_loss\":" << res.finalValLoss << ","
                  << "\"initial_perplexity\":" << res.initialPerplexity << ","
                  << "\"final_perplexity\":" << res.finalPerplexity << ","
                  << "\"initial_weight_norm\":" << res.initialWeightNorm << ","
                  << "\"final_weight_norm\":" << res.finalWeightNorm << ","
                  << "\"weight_delta_norm\":" << res.weightDeltaNorm << ","
                  << "\"weights_updated\":" << (res.weightsUpdated ? "true" : "false") << ","
                  << "\"checkpoint_path\":\"" << escapeJson(res.checkpointPath) << "\","
                  << "\"sample_generation\":\"" << escapeJson(res.sampleGeneration) << "\","
                  << "\"duration_s\":" << res.durationSeconds
                  << "}" << std::endl;
    }

    return res.success ? 0 : 1;
}

int runCheckpointValidation(const std::string& ckptPath, bool jsonOutput) {
    auto val = NexaNanoModel::validateCheckpoint(ckptPath);
    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (val.valid ? "ok" : "error") << "\","
                  << "\"valid\":" << (val.valid ? "true" : "false") << ","
                  << "\"version\":" << val.version << ","
                  << "\"arch_type\":" << val.archType << ","
                  << "\"vocab_size\":" << val.config.vocabSize << ","
                  << "\"dim\":" << val.config.dim << ","
                  << "\"intermediate_dim\":" << val.config.intermediateDim << ","
                  << "\"layers\":" << val.config.nLayers << ","
                  << "\"heads\":" << val.config.nHeads << ","
                  << "\"max_seq_len\":" << val.config.maxSeqLen << ","
                  << "\"rope_theta\":" << val.config.ropeTheta << ","
                  << "\"use_rope\":" << (val.config.useRope ? "true" : "false") << ","
                  << "\"tie_embeddings\":" << (val.config.tieEmbeddings ? "true" : "false") << ","
                  << "\"tokenizer_hash\":" << val.config.tokenizerHash << ","
                  << "\"tensors\":" << val.tensorCount << ","
                  << "\"weight_bytes\":" << val.totalWeightBytes << ","
                  << "\"step\":" << val.step << ","
                  << "\"loss\":" << val.loss << ","
                  << "\"has_optimizer_state\":" << (val.hasOptimizerState ? "true" : "false") << ","
                  << "\"error\":\"" << escapeJson(val.error) << "\""
                  << "}" << std::endl;
    } else {
        if (!val.valid) {
            std::cerr << "Checkpoint Invalid: " << val.error << std::endl;
            return 1;
        }
        std::cout << "Checkpoint Valid: " << ckptPath << std::endl;
        std::cout << "Format Version: " << val.version << " | Architecture Type: " << val.archType << std::endl;
        std::cout << "Vocab Size: " << val.config.vocabSize << " | Dim: " << val.config.dim
                  << " | FFN: " << val.config.intermediateDim << " | Ctx: " << val.config.maxSeqLen << std::endl;
        std::cout << "Layers: " << val.config.nLayers << " | Heads: " << val.config.nHeads
                  << " | RoPE: " << (val.config.useRope ? "true" : "false")
                  << " | Tied: " << (val.config.tieEmbeddings ? "true" : "false") << std::endl;
        std::cout << "Tensors: " << val.tensorCount << " | Total Weight Bytes: " << val.totalWeightBytes << std::endl;
        std::cout << "Training Step: " << val.step << " | Loss: " << val.loss << std::endl;
        std::cout << "Optimizer State: " << (val.hasOptimizerState ? "Present" : "None") << std::endl;
    }
    return (val.valid || jsonOutput) ? 0 : 1;
}

int runBpeTrain(
    const std::string& corpusPath,
    const std::string& outputPath,
    uint32_t targetVocab = 280,
    uint32_t minFreq = 2,
    bool jsonOutput = false
) {
    std::string corpus;
    if (std::filesystem::is_directory(corpusPath)) {
        for (const auto& entry : std::filesystem::directory_iterator(corpusPath)) {
            if (entry.path().extension() == ".txt") {
                std::ifstream ifs(entry.path(), std::ios::binary);
                if (ifs) {
                    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                    corpus += s + "\n";
                }
            }
        }
    } else {
        std::ifstream ifs(corpusPath, std::ios::binary);
        if (!ifs) {
            if (jsonOutput) {
                std::cout << "{\"status\":\"error\",\"error\":\"Cannot open corpus file: " << escapeJson(corpusPath) << "\"}" << std::endl;
            } else {
                std::cerr << "Error: Cannot open corpus file " << corpusPath << std::endl;
            }
            return 1;
        }
        corpus.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    }

    NexaBpeTokenizer bpe;
    if (corpus.empty()) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"Corpus is empty\"}" << std::endl;
        } else {
            std::cerr << "Error: Corpus is empty." << std::endl;
        }
        return 1;
    }
    bpe.train(corpus, targetVocab, minFreq);

    std::filesystem::path p(outputPath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    if (!bpe.save(outputPath)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"Cannot save BPE file: " << escapeJson(outputPath) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Cannot save BPE file " << outputPath << std::endl;
        }
        return 1;
    }

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"ok\","
                  << "\"vocab_size\":" << bpe.vocabSize() << ","
                  << "\"merges_count\":" << bpe.numMerges() << ","
                  << "\"output_path\":\"" << escapeJson(outputPath) << "\""
                  << "}" << std::endl;
    } else {
        std::cout << "BPE Training Successful!" << std::endl;
        std::cout << "Vocab Size: " << bpe.vocabSize() << " | Merges: " << bpe.numMerges() << std::endl;
        std::cout << "Saved To: " << outputPath << std::endl;
    }
    return 0;
}

int runQaTraining(
    const std::string& datasetDir = "data/datasets/qa",
    const std::string& ckptPath = "data/checkpoints/nexa_nano_qa.nexa",
    const std::string& nbtPath = "data/tokenizers/nexa_bpe_qa.nbt",
    uint32_t steps = 60,
    uint32_t batchSize = 2,
    uint32_t gradAccum = 1,
    float lr = 3e-3f,
    uint32_t warmup = 10,
    uint32_t seqLen = 32,
    bool jsonOutput = false
) {
    // 1. Train or load BPE tokenizer on QA corpus
    std::filesystem::path nbtParent = std::filesystem::path(nbtPath).parent_path();
    if (!nbtParent.empty()) {
        std::filesystem::create_directories(nbtParent);
    }
    NexaBpeTokenizer bpe;
    bool bpeLoaded = false;
    if (std::filesystem::exists(nbtPath)) {
        bpeLoaded = bpe.load(nbtPath);
    }
    if (!bpeLoaded) {
        std::string corpus;
        if (std::filesystem::is_directory(datasetDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(datasetDir)) {
                if (entry.path().extension() == ".txt") {
                    std::ifstream ifs(entry.path(), std::ios::binary);
                    if (ifs) {
                        std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                        corpus += s + "\n";
                    }
                }
            }
        } else if (std::filesystem::exists(datasetDir)) {
            std::ifstream ifs(datasetDir, std::ios::binary);
            if (ifs) {
                corpus.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            }
        }
        if (corpus.empty()) {
            if (jsonOutput) {
                std::cout << "{\"status\":\"error\",\"error\":\"Empty or missing dataset: " << escapeJson(datasetDir) << "\"}" << std::endl;
            } else {
                std::cerr << "Error: Empty or missing dataset: " << datasetDir << std::endl;
            }
            return 1;
        }
        bpe.train(corpus, 384, 2);
        bpe.save(nbtPath);
    }

    // 2. Load dataset with BPE tokenizer
    DatasetPipeline dataset;
    bool loaded = false;
    if (std::filesystem::is_directory(datasetDir)) {
        loaded = dataset.loadFromDirectory(datasetDir, bpe, 0.85f, seqLen);
    } else {
        loaded = dataset.loadFromFile(datasetDir, bpe, 0.85f, seqLen);
    }
    if (!loaded) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(dataset.metadata().errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error loading dataset: " << dataset.metadata().errorMessage << std::endl;
        }
        return 1;
    }

    // 3. Configure NEXA-1 Nano architecture
    NanoConfig cfg;
    cfg.vocabSize = bpe.vocabSize();
    cfg.dim = 64;
    cfg.intermediateDim = 128;
    cfg.nLayers = 2;
    cfg.nHeads = 2;
    cfg.maxSeqLen = seqLen * 2;

    NexaNanoModel model(cfg, 42);

    // 4. Configure pipeline training
    TrainingPipelineConfig pCfg;
    pCfg.numSteps = steps;
    pCfg.batchSize = batchSize;
    pCfg.gradAccumSteps = gradAccum;
    pCfg.learningRate = lr;
    pCfg.minLr = 1e-4f;
    pCfg.warmupSteps = warmup;
    pCfg.evalInterval = std::max(1u, steps / 6);
    pCfg.checkpointPath = ckptPath;
    pCfg.promptForSample = "Q: What is AI?\nA:";

    auto res = NexaTrainer::trainWithDatasetAdvanced(model, dataset, pCfg, &bpe);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (res.success ? "ok" : "error") << "\","
                  << "\"total_steps\":" << res.totalSteps << ","
                  << "\"initial_train_loss\":" << res.initialTrainLoss << ","
                  << "\"final_train_loss\":" << res.finalTrainLoss << ","
                  << "\"initial_val_loss\":" << res.initialValLoss << ","
                  << "\"final_val_loss\":" << res.finalValLoss << ","
                  << "\"initial_perplexity\":" << res.initialPerplexity << ","
                  << "\"final_perplexity\":" << res.finalPerplexity << ","
                  << "\"weight_delta_norm\":" << res.weightDeltaNorm << ","
                  << "\"checkpoint_path\":\"" << escapeJson(res.checkpointPath) << "\","
                  << "\"tokenizer_path\":\"" << escapeJson(nbtPath) << "\","
                  << "\"vocab_size\":" << bpe.vocabSize() << ","
                  << "\"sample_generation\":\"" << escapeJson(res.sampleGeneration) << "\","
                  << "\"duration_s\":" << res.durationSeconds
                  << "}" << std::endl;
    }

    return res.success ? 0 : 1;
}

int runQaAnswer(
    const std::string& ckptPath,
    const std::string& question,
    const std::string& nbtPath = "data/tokenizers/nexa_bpe_qa.nbt",
    uint32_t maxTokens = 48,
    float temp = 0.5f,
    uint64_t seed = 42,
    bool jsonOutput = false
) {
    auto val = NexaNanoModel::validateCheckpoint(ckptPath);
    if (!val.valid) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"Invalid checkpoint: " << escapeJson(val.error) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Invalid checkpoint: " << val.error << std::endl;
        }
        return 1;
    }

    NexaNanoModel model(val.config, 42);
    std::string loadErr;
    if (!model.loadCheckpoint(ckptPath, nullptr, nullptr, nullptr, &loadErr)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"Failed to load model: " << escapeJson(loadErr) << "\"}" << std::endl;
        } else {
            std::cerr << "Error loading checkpoint: " << loadErr << std::endl;
        }
        return 1;
    }

    std::unique_ptr<Tokenizer> tok;
    if (!nbtPath.empty()) {
        if (!std::filesystem::exists(nbtPath)) {
            std::string err = "Tokenizer file not found: " + nbtPath;
            if (jsonOutput) {
                std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
            } else {
                std::cerr << "Error: " << err << std::endl;
            }
            return 1;
        }
        auto bpe = std::make_unique<NexaBpeTokenizer>();
        if (!bpe->load(nbtPath)) {
            std::string err = "Failed to load BPE tokenizer: " + nbtPath;
            if (jsonOutput) {
                std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
            } else {
                std::cerr << "Error: " << err << std::endl;
            }
            return 1;
        }
        tok = std::move(bpe);
    } else if (val.config.vocabSize <= 260) {
        tok = std::make_unique<NexaByteTokenizer>();
    } else {
        std::string err = "Checkpoint requires a BPE .nbt tokenizer (vocab_size=" + std::to_string(val.config.vocabSize) + ")";
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: " << err << std::endl;
        }
        return 1;
    }

    if (!nbtPath.empty() && tok->vocabSize() != val.config.vocabSize) {
        std::string vErr = "Tokenizer vocabulary size (" + std::to_string(tok->vocabSize()) +
                           ") does not match checkpoint vocabulary size (" + std::to_string(val.config.vocabSize) + ")";
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(vErr) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: " << vErr << std::endl;
        }
        return 1;
    }

    std::string prompt = normalizeConversationPrompt(question);
    GenerationConfig genCfg;
    genCfg.maxNewTokens = maxTokens;
    genCfg.temperature = temp;
    genCfg.topK = 24;
    genCfg.topP = 0.9f;
    genCfg.repetitionPenalty = 1.15f;
    genCfg.seed = seed;
    genCfg.stopSequences = {
        "\nUser:", "\nAssistant:", "\nSystem:",
        "\nQ:", "\nQuestion:", "\n\n", "<|end_of_text|>"
    };

    auto genRes = model.generateText(prompt, *tok, genCfg);
    if (!genRes.success) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(genRes.errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error generating answer: " << genRes.errorMessage << std::endl;
        }
        return 1;
    }

    std::string cleanedAnswer = cleanAssistantReply(genRes.generatedText);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"ok\","
                  << "\"question\":\"" << escapeJson(question) << "\","
                  << "\"prompt\":\"" << escapeJson(prompt) << "\","
                  << "\"answer\":\"" << escapeJson(cleanedAnswer) << "\","
                  << "\"full_text\":\"" << escapeJson(genRes.fullText) << "\","
                  << "\"prompt_tokens\":" << genRes.promptTokens.size() << ","
                  << "\"tokens_generated\":" << genRes.tokensGenerated << ","
                  << "\"max_seq_len\":" << val.config.maxSeqLen << ","
                  << "\"vocab_size\":" << val.config.vocabSize << ","
                  << "\"use_rope\":" << (val.config.useRope ? "true" : "false") << ","
                  << "\"tie_embeddings\":" << (val.config.tieEmbeddings ? "true" : "false") << ","
                  << "\"duration_ms\":" << genRes.durationMs << ","
                  << "\"tokens_per_second\":" << genRes.tokensPerSecond << ","
                  << "\"finish_reason\":\"" << escapeJson(genRes.finishReason) << "\""
                  << "}" << std::endl;
    } else {
        std::cout << "Question: " << question << std::endl;
        std::cout << "Answer: " << cleanedAnswer << std::endl;
        std::cout << "(Prompt " << genRes.promptTokens.size() << "/" << val.config.maxSeqLen
                  << " tok, generated " << genRes.tokensGenerated << " tokens in "
                  << genRes.durationMs << " ms, " << std::fixed << std::setprecision(1)
                  << genRes.tokensPerSecond << " tok/s)" << std::endl;
    }
    return 0;
}

int runBenchmarkConfigs(uint32_t vocabSize = 768, bool jsonOutput = false) {
    auto entries = NexaTrainer::benchmarkConfigurations(vocabSize);
    if (jsonOutput) {
        std::cout << "{\"status\":\"ok\",\"vocab_size\":" << vocabSize << ",\"configs\":[";
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto& e = entries[i];
            std::cout << "{"
                      << "\"name\":\"" << escapeJson(e.name) << "\","
                      << "\"params\":" << e.totalParameters << ","
                      << "\"dim\":" << e.config.dim << ","
                      << "\"intermediate_dim\":" << e.config.intermediateDim << ","
                      << "\"layers\":" << e.config.nLayers << ","
                      << "\"heads\":" << e.config.nHeads << ","
                      << "\"context_length\":" << e.config.maxSeqLen << ","
                      << "\"weights_mb\":" << (e.memoryEstimate.weightsBytes / (1024.0 * 1024.0)) << ","
                      << "\"est_training_mb\":" << (e.memoryEstimate.totalTrainingBytes / (1024.0 * 1024.0)) << ","
                      << "\"est_inference_mb\":" << (e.memoryEstimate.totalInferenceBytes / (1024.0 * 1024.0)) << ","
                      << "\"avg_step_ms\":" << e.avgStepMs << ","
                      << "\"tokens_per_sec\":" << e.tokensPerSec << ","
                      << "\"peak_rss_mb\":" << e.peakRssMb << ","
                      << "\"feasible\":" << (e.feasibleFor4GbCpu ? "true" : "false")
                      << "}" << (i + 1 < entries.size() ? "," : "");
        }
        std::cout << "]}" << std::endl;
    } else {
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << "                   NEXA-1 MODEL CAPACITY & HARDWARE BENCHMARK (CPU)                       " << std::endl;
        std::cout << "==========================================================================================" << std::endl;
        std::cout << std::left << std::setw(16) << "Config"
                  << std::right << std::setw(10) << "Params"
                  << std::setw(6) << "Dim"
                  << std::setw(5) << "L"
                  << std::setw(5) << "H"
                  << std::setw(6) << "Ctx"
                  << std::setw(10) << "Wts(MB)"
                  << std::setw(11) << "Train(MB)"
                  << std::setw(10) << "ms/step"
                  << std::setw(10) << "tok/s"
                  << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;
        for (const auto& e : entries) {
            std::cout << std::left << std::setw(16) << e.name
                      << std::right << std::setw(10) << e.totalParameters
                      << std::setw(6) << e.config.dim
                      << std::setw(5) << e.config.nLayers
                      << std::setw(5) << e.config.nHeads
                      << std::setw(6) << e.config.maxSeqLen
                      << std::fixed << std::setprecision(2)
                      << std::setw(10) << (e.memoryEstimate.weightsBytes / (1024.0 * 1024.0))
                      << std::setw(11) << (e.memoryEstimate.totalTrainingBytes / (1024.0 * 1024.0))
                      << std::setprecision(1)
                      << std::setw(10) << e.avgStepMs
                      << std::setw(10) << e.tokensPerSec
                      << (e.feasibleFor4GbCpu ? "  [OK]" : "  [SLOW]")
                      << std::endl;
        }
        std::cout << "Peak Process RSS: " << std::fixed << std::setprecision(2) << NexaTrainer::getPeakRssMb() << " MB" << std::endl;
        std::cout << "==========================================================================================\n" << std::endl;
    }
    return 0;
}

int runPhase35Training(
    const std::string& datasetDir = "data/datasets/phase35/train_val",
    const std::string& bestCkptPath = "data/checkpoints/nexa_phase35_best.nexa",
    const std::string& nbtPath = "data/tokenizers/nexa_bpe_phase35.nbt",
    uint32_t steps = 250,
    uint32_t dim = 128,
    uint32_t layers = 4,
    uint32_t heads = 4,
    uint32_t targetVocab = 768,
    uint32_t seqLen = 64,
    uint32_t batchSize = 2,
    uint32_t gradAccum = 2,
    float lr = 2.5e-3f,
    uint64_t seed = 42,
    bool jsonOutput = false
) {
    namespace fs = std::filesystem;
    if (!fs::exists(datasetDir) || !fs::is_directory(datasetDir)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"Missing dataset directory: " << escapeJson(datasetDir) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Missing dataset directory " << datasetDir << std::endl;
        }
        return 1;
    }

    fs::path nbtParent = fs::path(nbtPath).parent_path();
    if (!nbtParent.empty()) fs::create_directories(nbtParent);

    // 1. Train BPE tokenizer cleanly on dataset directory (always retrain if missing or requested vocab differs)
    NexaBpeTokenizer bpe;
    bool needTrainBpe = true;
    if (fs::exists(nbtPath) && bpe.load(nbtPath) && bpe.vocabSize() == targetVocab) {
        needTrainBpe = false;
    }
    if (needTrainBpe) {
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(datasetDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        std::string corpus;
        for (const auto& f : files) {
            std::ifstream ifs(f, std::ios::binary);
            if (ifs) {
                corpus.append((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                corpus += "\n";
            }
        }
        bpe.train(corpus, targetVocab, 2);
        bpe.save(nbtPath);
    }

    // 2. Load instruction-tuning dataset with prompt loss masking and shuffled example split
    DatasetPipeline dataset;
    if (!dataset.loadInstructionExamplesFromDirectory(datasetDir, bpe, 0.85f, seqLen, true, seed)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(dataset.metadata().errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error loading dataset: " << dataset.metadata().errorMessage << std::endl;
        }
        return 1;
    }

    // 3. Configure chosen model architecture
    NanoConfig cfg;
    cfg.vocabSize = static_cast<uint32_t>(bpe.vocabSize());
    cfg.dim = dim;
    cfg.intermediateDim = dim * 2;
    cfg.nLayers = layers;
    cfg.nHeads = heads;
    cfg.maxSeqLen = seqLen;

    NexaNanoModel model(cfg, seed);

    std::string finalCkptPath = bestCkptPath;
    size_t dotPos = finalCkptPath.rfind(".nexa");
    if (dotPos != std::string::npos) {
        finalCkptPath = finalCkptPath.substr(0, dotPos) + "_final.nexa";
    } else {
        finalCkptPath += "_final";
    }

    TrainingPipelineConfig pCfg;
    pCfg.numSteps = steps;
    pCfg.batchSize = batchSize;
    pCfg.gradAccumSteps = gradAccum;
    pCfg.learningRate = lr;
    pCfg.minLr = 1e-4f;
    pCfg.warmupSteps = std::max(5u, steps / 10);
    pCfg.evalInterval = std::max(5u, steps / 10);
    pCfg.earlyStopPatience = 6;
    pCfg.weightDecay = 0.05f;
    pCfg.seed = seed;
    pCfg.shuffleBatches = true;
    pCfg.ignoreIndex = DatasetPipeline::IGNORE_INDEX;
    pCfg.checkpointPath = finalCkptPath;
    pCfg.bestCheckpointPath = bestCkptPath;
    pCfg.promptForSample = "User: What is a CPU?\nAssistant:";

    auto res = NexaTrainer::trainWithDatasetAdvanced(model, dataset, pCfg, &bpe);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (res.success ? "ok" : "error") << "\","
                  << "\"total_parameters\":" << model.totalParameters() << ","
                  << "\"vocab_size\":" << bpe.vocabSize() << ","
                  << "\"dim\":" << cfg.dim << ","
                  << "\"layers\":" << cfg.nLayers << ","
                  << "\"heads\":" << cfg.nHeads << ","
                  << "\"seq_len\":" << cfg.maxSeqLen << ","
                  << "\"examples\":" << dataset.metadata().exampleCount << ","
                  << "\"total_tokens\":" << dataset.metadata().totalTokens << ","
                  << "\"masked_target_tokens\":" << dataset.metadata().maskedTokensCount << ","
                  << "\"unmasked_target_tokens\":" << dataset.metadata().unmaskedTargetTokensCount << ","
                  << "\"total_steps\":" << res.totalSteps << ","
                  << "\"initial_train_loss\":" << res.initialTrainLoss << ","
                  << "\"final_train_loss\":" << res.finalTrainLoss << ","
                  << "\"initial_val_loss\":" << res.initialValLoss << ","
                  << "\"final_val_loss\":" << res.finalValLoss << ","
                  << "\"best_val_loss\":" << res.bestValLoss << ","
                  << "\"best_checkpoint_step\":" << res.bestCheckpointStep << ","
                  << "\"best_perplexity\":" << res.bestPerplexity << ","
                  << "\"overfitting_detected\":" << (res.overfittingDetected ? "true" : "false") << ","
                  << "\"tokens_per_sec\":" << res.tokensPerSecond << ","
                  << "\"peak_rss_mb\":" << res.peakRssMb << ","
                  << "\"duration_s\":" << res.durationSeconds << ","
                  << "\"best_checkpoint_path\":\"" << escapeJson(res.bestCheckpointPath) << "\","
                  << "\"tokenizer_path\":\"" << escapeJson(nbtPath) << "\","
                  << "\"sample_generation\":\"" << escapeJson(res.sampleGeneration) << "\""
                  << "}" << std::endl;
    }

    return res.success ? 0 : 1;
}

int runUnseenEvaluation(
    const std::string& ckptPath = "data/checkpoints/nexa_phase35_best.nexa",
    const std::string& nbtPath = "data/tokenizers/nexa_bpe_phase35.nbt",
    const std::string& unseenPath = "data/datasets/phase35/unseen_test_qa.txt",
    const std::string& trainValDir = "data/datasets/phase35/train_val",
    uint32_t maxTokens = 48,
    float temp = 0.4f,
    uint64_t seed = 42,
    bool jsonOutput = false
) {
    auto val = NexaNanoModel::validateCheckpoint(ckptPath);
    if (!val.valid) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(val.error) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Invalid checkpoint: " << val.error << std::endl;
        }
        return 1;
    }

    NexaNanoModel model(val.config, seed);
    std::string loadErr;
    if (!model.loadCheckpoint(ckptPath, nullptr, nullptr, nullptr, &loadErr)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(loadErr) << "\"}" << std::endl;
        } else {
            std::cerr << "Error loading checkpoint: " << loadErr << std::endl;
        }
        return 1;
    }

    NexaBpeTokenizer bpe;
    if (!bpe.load(nbtPath) || bpe.vocabSize() != val.config.vocabSize) {
        std::string err = "Failed to load compatible .nbt tokenizer from " + nbtPath;
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(err) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: " << err << std::endl;
        }
        return 1;
    }

    auto summary = NexaTrainer::evaluateUnseenQuestions(model, bpe, unseenPath, trainValDir, maxTokens, temp, seed);
    if (!summary.success) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(summary.errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error evaluating unseen questions: " << summary.errorMessage << std::endl;
        }
        return 1;
    }

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"ok\","
                  << "\"checkpoint\":\"" << escapeJson(ckptPath) << "\","
                  << "\"total_parameters\":" << model.totalParameters() << ","
                  << "\"disjoint_verified\":" << (summary.disjointVerified ? "true" : "false") << ","
                  << "\"total_questions\":" << summary.totalQuestions << ","
                  << "\"non_empty_responses\":" << summary.nonEmptyResponses << ","
                  << "\"clean_stop_count\":" << summary.cleanStopCount << ","
                  << "\"relevance_hit_count\":" << summary.relevanceHitCount << ","
                  << "\"uncertainty_questions\":" << summary.uncertaintyQuestionCount << ","
                  << "\"uncertainty_matched\":" << summary.uncertaintyMatchedCount << ","
                  << "\"avg_relevance_score\":" << summary.avgRelevanceScore << ","
                  << "\"avg_coherence_score\":" << summary.avgCoherenceScore << ","
                  << "\"avg_factual_accuracy_score\":" << summary.avgFactualAccuracyScore << ","
                  << "\"avg_instruction_following_score\":" << summary.avgInstructionFollowingScore << ","
                  << "\"avg_latency_ms\":" << summary.avgLatencyMs << ","
                  << "\"avg_tokens_per_sec\":" << summary.avgTokensPerSec << ","
                  << "\"avg_repetition_rate\":" << summary.avgRepetitionRate << ","
                  << "\"avg_word_repetition_rate\":" << summary.avgWordRepetitionRate << ","
                  << "\"peak_rss_mb\":" << summary.peakRssMb << ","
                  << "\"samples\":[";
        for (size_t i = 0; i < summary.results.size(); ++i) {
            const auto& r = summary.results[i];
            std::cout << "{"
                      << "\"category\":\"" << escapeJson(r.category) << "\","
                      << "\"expected_behavior\":\"" << escapeJson(r.expectedBehavior) << "\","
                      << "\"question\":\"" << escapeJson(r.question) << "\","
                      << "\"answer\":\"" << escapeJson(r.generatedAnswer) << "\","
                      << "\"tokens_generated\":" << r.tokensGenerated << ","
                      << "\"duration_ms\":" << r.durationMs << ","
                      << "\"repetition_rate\":" << r.trigramRepetitionRate << ","
                      << "\"word_repetition_rate\":" << r.wordRepetitionRate << ","
                      << "\"relevance_score\":" << r.relevanceScore << ","
                      << "\"coherence_score\":" << r.coherenceScore << ","
                      << "\"factual_accuracy_score\":" << r.factualAccuracyScore << ","
                      << "\"instruction_following_score\":" << r.instructionFollowingScore << ","
                      << "\"clean_stop\":" << (r.cleanStop ? "true" : "false") << ","
                      << "\"relevance_hit\":" << (r.relevanceHit ? "true" : "false")
                      << "}" << (i + 1 < summary.results.size() ? "," : "");
        }
        std::cout << "]}" << std::endl;
    } else {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "      NEXA UNSEEN QUESTION EVALUATION (" << summary.totalQuestions << " QUESTIONS)           " << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "Checkpoint:          " << ckptPath << " (" << model.totalParameters() << " params)" << std::endl;
        std::cout << "Disjoint Verified:   " << (summary.disjointVerified ? "YES (0 overlap with train_val)" : "NO") << std::endl;
        std::cout << "Non-empty/Valid UTF8:" << " " << summary.nonEmptyResponses << "/" << summary.totalQuestions << std::endl;
        std::cout << "Clean Stop (<eos>):  " << summary.cleanStopCount << "/" << summary.totalQuestions << std::endl;
        std::cout << "Keyword/Relevance:   " << summary.relevanceHitCount << "/" << summary.totalQuestions
                  << " (avg score " << std::fixed << std::setprecision(3) << summary.avgRelevanceScore << ")" << std::endl;
        std::cout << "Coherence Score:     " << summary.avgCoherenceScore << " (English lexicon + structure)" << std::endl;
        std::cout << "Factual Accuracy:    " << summary.avgFactualAccuracyScore << std::endl;
        std::cout << "Instruction Follow:  " << summary.avgInstructionFollowingScore << std::endl;
        std::cout << "Uncertainty Refusal: " << summary.uncertaintyMatchedCount << "/" << summary.uncertaintyQuestionCount << std::endl;
        std::cout << "Avg Latency:         " << std::setprecision(2) << summary.avgLatencyMs << " ms ("
                  << std::setprecision(1) << summary.avgTokensPerSec << " tok/s)" << std::endl;
        std::cout << "Avg Repetition:      3-gram=" << std::setprecision(3) << summary.avgRepetitionRate
                  << " | word=" << summary.avgWordRepetitionRate << std::endl;
        std::cout << "Peak Process RSS:    " << std::setprecision(2) << summary.peakRssMb << " MB" << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;
        for (size_t i = 0; i < summary.results.size(); ++i) {
            const auto& r = summary.results[i];
            std::cout << "[" << (i + 1) << "|" << r.category << "] Q: " << r.question << std::endl;
            std::cout << "    A: " << r.generatedAnswer
                      << " (tok=" << r.tokensGenerated
                      << ", rel=" << std::setprecision(2) << r.relevanceScore
                      << ", coh=" << r.coherenceScore
                      << ", fact=" << r.factualAccuracyScore
                      << ", inst=" << r.instructionFollowingScore << ")" << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }
    return 0;
}

int runDatasetAudit(
    const std::string& datasetPath,
    const std::string& unseenPath = "",
    const std::string& nbtPath = "",
    uint32_t targetVocab = 768,
    bool requireVerifiedLicense = false,
    bool jsonOutput = false
) {
    namespace fs = std::filesystem;
    NexaBpeTokenizer bpe;
    bool usingTrainedBpe = false;

    if (!nbtPath.empty() && fs::exists(nbtPath) && bpe.load(nbtPath)) {
        usingTrainedBpe = true;
    } else {
        std::string corpus;
        std::string scanDir = datasetPath;
        if (fs::is_regular_file(datasetPath)) {
            scanDir = fs::path(datasetPath).parent_path().string();
        }
        std::string trainSub = (fs::path(scanDir) / "train_val").string();
        if (fs::is_directory(trainSub)) scanDir = trainSub;
        if (fs::is_directory(scanDir)) {
            std::vector<std::string> files;
            for (const auto& entry : fs::directory_iterator(scanDir)) {
                if (entry.is_regular_file()) {
                    auto ext = entry.path().extension();
                    if (ext == ".txt" || ext == ".jsonl") files.push_back(entry.path().string());
                }
            }
            std::sort(files.begin(), files.end());
            for (const auto& f : files) {
                std::ifstream ifs(f, std::ios::binary);
                if (ifs) corpus.append((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                corpus += "\n";
            }
        }
        if (!corpus.empty()) {
            bpe.train(corpus, targetVocab, 2);
            usingTrainedBpe = true;
        }
    }

    NexaByteTokenizer fallbackTok;
    Tokenizer& activeTok = usingTrainedBpe ? static_cast<Tokenizer&>(bpe) : static_cast<Tokenizer&>(fallbackTok);

    DatasetPipeline ds;
    bool ok = ds.loadManagedDataset(
        datasetPath, activeTok, 0.85f, 64, true, 42, unseenPath, requireVerifiedLicense, false
    );
    const auto& meta = ds.metadata();
    const auto& rep = ds.qualityReport();
    bool passedAll = ok && meta.contaminationCount == 0 && (!requireVerifiedLicense || meta.licenseVerified);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (passedAll ? "ok" : "error") << "\","
                  << "\"valid\":" << (ok ? "true" : "false") << ","
                  << "\"dataset_version\":\"" << escapeJson(meta.datasetVersion) << "\","
                  << "\"content_hash\":\"" << escapeJson(meta.contentHashHex) << "\","
                  << "\"source\":\"" << escapeJson(meta.source) << "\","
                  << "\"license\":\"" << escapeJson(meta.license) << "\","
                  << "\"license_verified\":" << (meta.licenseVerified ? "true" : "false") << ","
                  << "\"documents\":" << meta.documentsCount << ","
                  << "\"examples\":" << meta.exampleCount << ","
                  << "\"train_examples\":" << meta.trainExamplesCount << ","
                  << "\"val_examples\":" << meta.valExamplesCount << ","
                  << "\"bytes\":" << meta.fileSizeBytes << ","
                  << "\"total_tokens\":" << meta.totalTokens << ","
                  << "\"unique_tokens\":" << meta.uniqueTokensCount << ","
                  << "\"train_tokens\":" << meta.trainTokensCount << ","
                  << "\"val_tokens\":" << meta.valTokensCount << ","
                  << "\"masked_tokens\":" << meta.maskedTokensCount << ","
                  << "\"unmasked_tokens\":" << meta.unmaskedTargetTokensCount << ","
                  << "\"exact_duplicates_removed\":" << meta.exactDuplicatesRemoved << ","
                  << "\"near_duplicates_removed\":" << meta.nearDuplicatesRemoved << ","
                  << "\"quality_issues_removed\":" << meta.qualityIssuesRemoved << ","
                  << "\"contamination_count\":" << meta.contaminationCount << ","
                  << "\"duplication_rate\":" << rep.duplicationRate << ","
                  << "\"quality_gate_passed\":" << (rep.passedQualityGate ? "true" : "false") << ","
                  << "\"error\":\"" << escapeJson(meta.errorMessage) << "\""
                  << "}" << std::endl;
    } else {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "               NEXA DATASET AUDIT & QUALITY REPORT              " << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "Path:              " << datasetPath << std::endl;
        std::cout << "Version / Hash:    " << meta.datasetVersion << " (" << meta.contentHashHex << ")" << std::endl;
        std::cout << "Source:            " << meta.source << std::endl;
        std::cout << "License:           " << meta.license << " (verified=" << (meta.licenseVerified ? "YES" : "NO") << ")" << std::endl;
        std::cout << "Documents / Bytes: " << meta.documentsCount << " files, " << meta.fileSizeBytes << " bytes" << std::endl;
        std::cout << "Examples (Tr/Val): " << meta.exampleCount << " (" << meta.trainExamplesCount << " / " << meta.valExamplesCount << ")" << std::endl;
        std::cout << "Tokens (Total/Unq):" << " " << meta.totalTokens << " / " << meta.uniqueTokensCount
                  << " (masked=" << meta.maskedTokensCount << ", active=" << meta.unmaskedTargetTokensCount << ")" << std::endl;
        std::cout << "Duplicates Pruned: exact=" << meta.exactDuplicatesRemoved << ", near=" << meta.nearDuplicatesRemoved << std::endl;
        std::cout << "Quality Rejected:  " << meta.qualityIssuesRemoved
                  << " (utf8=" << rep.utf8ErrorCount << ", incomp=" << rep.incompleteSentenceCount
                  << ", rep=" << rep.repetitiveExampleCount << ", contra=" << rep.contradictoryAnswerCount
                  << ", low=" << rep.lowInformationCount << ", fmt=" << rep.formattingErrorCount
                  << ", conv=" << rep.malformedConversationCount << ")" << std::endl;
        std::cout << "Contamination:     " << meta.contaminationCount << " matches" << std::endl;
        if (!meta.errorMessage.empty()) {
            std::cout << "Error:             " << meta.errorMessage << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }
    return passedAll ? 0 : 1;
}

int runPhase36Training(
    const std::string& datasetPath,
    const std::string& bestCkptPath,
    const std::string& nbtPath,
    const std::string& unseenPath,
    const std::string& experimentId,
    uint32_t steps = 200,
    uint32_t dim = 64,
    uint32_t layers = 2,
    uint32_t heads = 2,
    uint32_t targetVocab = 768,
    uint32_t seqLen = 64,
    uint32_t batchSize = 2,
    uint32_t gradAccum = 2,
    float lr = 2.5e-3f,
    uint32_t earlyStopPatience = 6,
    uint64_t seed = 42,
    bool requireVerifiedLicense = false,
    bool jsonOutput = false,
    bool useRope = false,
    bool tieEmbeddings = false,
    uint32_t intermediateDim = 0,
    const std::string& resumeCkptPath = "",
    float minLr = 1e-4f
) {
    namespace fs = std::filesystem;
    if (!fs::exists(datasetPath)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"Missing dataset path: " << escapeJson(datasetPath) << "\"}" << std::endl;
        } else {
            std::cerr << "Error: Missing dataset path " << datasetPath << std::endl;
        }
        return 1;
    }

    fs::path nbtParent = fs::path(nbtPath).parent_path();
    if (!nbtParent.empty()) fs::create_directories(nbtParent);

    // 1. Load or train BPE tokenizer strictly on training files (never unseen test file)
    NexaBpeTokenizer bpe;
    bool needTrainBpe = true;
    if (fs::exists(nbtPath) && bpe.load(nbtPath) && bpe.vocabSize() == targetVocab) {
        needTrainBpe = false;
    }
    if (needTrainBpe) {
        std::string scanDir = fs::is_regular_file(datasetPath) ? fs::path(datasetPath).parent_path().string() : datasetPath;
        std::string trainSub = (fs::path(scanDir) / "train_val").string();
        if (fs::is_directory(trainSub)) scanDir = trainSub;

        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(scanDir)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension();
                if (ext == ".txt" || ext == ".jsonl") files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        std::string corpus;
        for (const auto& f : files) {
            std::ifstream ifs(f, std::ios::binary);
            if (ifs) {
                corpus.append((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                corpus += "\n";
            }
        }
        bpe.train(corpus, targetVocab, 2);
        bpe.save(nbtPath);
    }

    // 2. Load managed dataset with quality checks, contamination check, and reproducible split
    DatasetPipeline dataset;
    if (!dataset.loadManagedDataset(datasetPath, bpe, 0.85f, seqLen, true, seed, unseenPath, requireVerifiedLicense, true)) {
        if (jsonOutput) {
            std::cout << "{\"status\":\"error\",\"error\":\"" << escapeJson(dataset.metadata().errorMessage) << "\"}" << std::endl;
        } else {
            std::cerr << "Error loading managed dataset: " << dataset.metadata().errorMessage << std::endl;
        }
        return 1;
    }

    // 3. Configure model architecture
    NanoConfig cfg;
    cfg.vocabSize = static_cast<uint32_t>(bpe.vocabSize());
    cfg.dim = dim;
    cfg.intermediateDim = intermediateDim > 0 ? intermediateDim : dim * 2;
    cfg.nLayers = layers;
    cfg.nHeads = heads;
    cfg.maxSeqLen = seqLen;
    cfg.useRope = useRope;
    cfg.tieEmbeddings = tieEmbeddings;

    std::string tokHashHex = ExperimentManager::computeFileHashHex(nbtPath);
    if (tokHashHex.size() >= 16) {
        try {
            cfg.tokenizerHash = std::stoull(tokHashHex.substr(0, 16), nullptr, 16);
        } catch (...) {
            cfg.tokenizerHash = 0;
        }
    }

    NexaNanoModel model(cfg, seed);

    std::string finalCkptPath = bestCkptPath;
    size_t dotPos = finalCkptPath.rfind(".nexa");
    if (dotPos != std::string::npos) {
        finalCkptPath = finalCkptPath.substr(0, dotPos) + "_final.nexa";
    } else {
        finalCkptPath += "_final";
    }

    TrainingPipelineConfig pCfg;
    pCfg.numSteps = steps;
    pCfg.batchSize = batchSize;
    pCfg.gradAccumSteps = gradAccum;
    pCfg.learningRate = lr;
    pCfg.minLr = minLr;
    pCfg.warmupSteps = std::max(5u, steps / 10);
    pCfg.evalInterval = std::max(5u, steps / 10);
    pCfg.earlyStopPatience = earlyStopPatience;
    pCfg.weightDecay = 0.05f;
    pCfg.seed = seed;
    pCfg.shuffleBatches = true;
    pCfg.ignoreIndex = DatasetPipeline::IGNORE_INDEX;
    pCfg.checkpointPath = finalCkptPath;
    pCfg.bestCheckpointPath = bestCkptPath;
    pCfg.resumeFromCheckpoint = resumeCkptPath;
    pCfg.promptForSample = "User: What is a CPU?\nAssistant:";

    auto res = NexaTrainer::trainWithDatasetAdvanced(model, dataset, pCfg, &bpe);

    // 4. Evaluate best checkpoint on unseen questions if unseenPath was supplied
    ExperimentEvalSummary expEval;
    if (res.success && !unseenPath.empty() && fs::exists(unseenPath) && fs::exists(res.bestCheckpointPath)) {
        NexaNanoModel bestModel(cfg, seed);
        if (bestModel.loadCheckpoint(res.bestCheckpointPath)) {
            std::string trainValDir = fs::is_regular_file(datasetPath) ? fs::path(datasetPath).parent_path().string() : datasetPath;
            if (fs::is_directory(fs::path(trainValDir) / "train_val")) {
                trainValDir = (fs::path(trainValDir) / "train_val").string();
            }
            auto evalSum = NexaTrainer::evaluateUnseenQuestions(bestModel, bpe, unseenPath, trainValDir, 40, 0.4f, seed);
            expEval.evaluated = evalSum.success;
            expEval.unseenSuitePath = unseenPath;
            expEval.disjointVerified = evalSum.disjointVerified;
            expEval.totalQuestions = evalSum.totalQuestions;
            expEval.relevanceHits = evalSum.relevanceHitCount;
            expEval.avgCoherenceScore = evalSum.avgCoherenceScore;
            expEval.avgFactualScore = evalSum.avgFactualAccuracyScore;
            expEval.avgInstructionScore = evalSum.avgInstructionFollowingScore;
            expEval.avgRepetitionRate = evalSum.avgRepetitionRate;
            expEval.cleanStopCount = evalSum.cleanStopCount;
            expEval.uncertaintyMatched = evalSum.uncertaintyMatchedCount;
            expEval.uncertaintyQuestions = evalSum.uncertaintyQuestionCount;
            expEval.avgLatencyMs = evalSum.avgLatencyMs;
        }
    }

    // 5. Save structured ExperimentRecord JSON
    ExperimentRecord expRec;
    expRec.experimentId = experimentId.empty() ? ("exp_" + dataset.metadata().datasetVersion) : experimentId;
    expRec.timestampIso = ExperimentManager::currentIsoTimestamp();
    expRec.datasetLabel = datasetPath;
    expRec.modelConfig = cfg;
    expRec.totalParameters = model.totalParameters();
    expRec.archPrep = NexaNanoModel::getArchitecturePrepSpec();
    expRec.datasetMeta = dataset.metadata();
    expRec.qualityReport = dataset.qualityReport();
    expRec.tokenizerInfo.type = "bpe";
    expRec.tokenizerInfo.path = nbtPath;
    expRec.tokenizerInfo.versionHash = tokHashHex;
    expRec.tokenizerInfo.vocabSize = static_cast<uint32_t>(bpe.vocabSize());
    expRec.tokenizerInfo.numMerges = static_cast<uint32_t>(bpe.numMerges());
    expRec.trainingConfig = pCfg;
    expRec.trainingResult = res;
    expRec.evalSummary = expEval;

    std::string expJsonPath;
    ExperimentManager::saveExperimentRecord(expRec, "data/experiments", &expJsonPath);

    if (jsonOutput) {
        std::cout << "{"
                  << "\"status\":\"" << (res.success ? "ok" : "error") << "\","
                  << "\"experiment_id\":\"" << escapeJson(expRec.experimentId) << "\","
                  << "\"experiment_path\":\"" << escapeJson(expJsonPath) << "\","
                  << "\"dataset_version\":\"" << escapeJson(dataset.metadata().datasetVersion) << "\","
                  << "\"license\":\"" << escapeJson(dataset.metadata().license) << "\","
                  << "\"license_verified\":" << (dataset.metadata().licenseVerified ? "true" : "false") << ","
                  << "\"total_parameters\":" << model.totalParameters() << ","
                  << "\"vocab_size\":" << bpe.vocabSize() << ","
                  << "\"intermediate_dim\":" << cfg.intermediateDim << ","
                  << "\"max_seq_len\":" << cfg.maxSeqLen << ","
                  << "\"use_rope\":" << (cfg.useRope ? "true" : "false") << ","
                  << "\"tie_embeddings\":" << (cfg.tieEmbeddings ? "true" : "false") << ","
                  << "\"documents\":" << dataset.metadata().documentsCount << ","
                  << "\"examples\":" << dataset.metadata().exampleCount << ","
                  << "\"bytes\":" << dataset.metadata().fileSizeBytes << ","
                  << "\"total_tokens\":" << dataset.metadata().totalTokens << ","
                  << "\"unique_tokens\":" << dataset.metadata().uniqueTokensCount << ","
                  << "\"masked_tokens\":" << dataset.metadata().maskedTokensCount << ","
                  << "\"unmasked_tokens\":" << dataset.metadata().unmaskedTargetTokensCount << ","
                  << "\"contamination_count\":" << dataset.metadata().contaminationCount << ","
                  << "\"total_steps\":" << res.totalSteps << ","
                  << "\"initial_train_loss\":" << res.initialTrainLoss << ","
                  << "\"final_train_loss\":" << res.finalTrainLoss << ","
                  << "\"initial_val_loss\":" << res.initialValLoss << ","
                  << "\"final_val_loss\":" << res.finalValLoss << ","
                  << "\"best_val_loss\":" << res.bestValLoss << ","
                  << "\"best_checkpoint_step\":" << res.bestCheckpointStep << ","
                  << "\"best_perplexity\":" << res.bestPerplexity << ","
                  << "\"overfitting_detected\":" << (res.overfittingDetected ? "true" : "false") << ","
                  << "\"tokens_per_sec\":" << res.tokensPerSecond << ","
                  << "\"peak_rss_mb\":" << res.peakRssMb << ","
                  << "\"duration_s\":" << res.durationSeconds << ","
                  << "\"best_checkpoint_path\":\"" << escapeJson(res.bestCheckpointPath) << "\","
                  << "\"tokenizer_path\":\"" << escapeJson(nbtPath) << "\","
                  << "\"eval_evaluated\":" << (expEval.evaluated ? "true" : "false") << ","
                  << "\"eval_questions\":" << expEval.totalQuestions << ","
                  << "\"eval_relevance_hits\":" << expEval.relevanceHits << ","
                  << "\"eval_coherence\":" << expEval.avgCoherenceScore << ","
                  << "\"eval_factual\":" << expEval.avgFactualScore << ","
                  << "\"eval_instruction\":" << expEval.avgInstructionScore << ","
                  << "\"eval_repetition\":" << expEval.avgRepetitionRate
                  << "}" << std::endl;
    }

    return res.success ? 0 : 1;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string arg = argv[1];

        // Machine-readable JSON endpoints for Node.js process bridge
        if (arg == "--version" || arg == "--info-json") {
            std::cout << "{"
                      << "\"engine\":\"nexa-native-runtime\","
                      << "\"version\":\"1.0.0-cxx20\","
                      << "\"standard\":\"C++20\","
                      << "\"arch\":\"" <<
#if defined(__x86_64__) || defined(_M_X64)
                         "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
                         "arm64"
#else
                         "generic"
#endif
                      << "\","
                      << "\"status\":\"ready\""
                      << "}" << std::endl;
            return 0;
        }

        if (arg == "--test-tensor-json") {
            return runTensorTestJson();
        }

        if (arg == "--forward-pass-json") {
            uint32_t dim = 128;
            uint32_t seqLen = 8;
            if (argc > 2) dim = std::stoul(argv[2]);
            if (argc > 3) seqLen = std::stoul(argv[3]);
            return runForwardPassJson(dim, seqLen);
        }

        if (arg == "--tokenize-json") {
            if (argc < 3) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Missing text argument for tokenize\"}" << std::endl;
                return 1;
            }
            std::string nbtPath = argc > 3 ? argv[3] : "";
            std::unique_ptr<Tokenizer> tok;
            std::string tokType = "byte";
            if (!nbtPath.empty()) {
                if (!std::filesystem::exists(nbtPath)) {
                    std::cout << "{\"status\":\"error\",\"error\":\"Tokenizer file not found: " << escapeJson(nbtPath) << "\"}" << std::endl;
                    return 1;
                }
                auto bpe = std::make_unique<NexaBpeTokenizer>();
                if (!bpe->load(nbtPath)) {
                    std::cout << "{\"status\":\"error\",\"error\":\"Failed to load BPE tokenizer: " << escapeJson(nbtPath) << "\"}" << std::endl;
                    return 1;
                }
                tok = std::move(bpe);
                tokType = "bpe";
            } else {
                tok = std::make_unique<NexaByteTokenizer>();
            }
            const bool addBos = nbtPath.empty();
            auto tokens = tok->encode(argv[2], addBos, false);
            std::cout << "{\"status\":\"ok\",\"tokenizer\":\"" << tokType << "\",\"vocab_size\":" << tok->vocabSize() << ",\"tokens\":[";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i] << (i + 1 < tokens.size() ? "," : "");
            }
            std::cout << "],\"count\":" << tokens.size() << "}" << std::endl;
            return 0;
        }

        if (arg == "--inspect-json") {
            if (argc < 3) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Missing path argument for inspect\"}" << std::endl;
                return 1;
            }
            auto val = NexaNanoModel::validateCheckpoint(argv[2]);
            auto meta = ModelLoader::inspectModel(argv[2]);
            if (!val.valid && !meta.isValid) {
                std::cerr << "{\"status\":\"error\",\"error\":\"File is not a valid .nexa model format\"}" << std::endl;
                return 1;
            }
            uint32_t version = val.valid ? val.version : meta.header.version;
            uint32_t dim = val.valid ? val.config.dim : meta.header.dim;
            uint32_t interDim = val.valid ? val.config.intermediateDim : meta.header.intermediateDim;
            uint32_t layers = val.valid ? val.config.nLayers : meta.header.nLayers;
            uint32_t heads = val.valid ? val.config.nHeads : meta.header.nHeads;
            uint32_t kvHeads = meta.isValid ? meta.header.nKvHeads : heads;
            uint32_t vocabSize = val.valid ? val.config.vocabSize : meta.header.vocabSize;
            uint32_t maxSeqLen = val.valid ? val.config.maxSeqLen : meta.header.maxSeqLen;
            float ropeTheta = val.valid ? val.config.ropeTheta : meta.header.ropeTheta;
            bool useRope = val.valid ? val.config.useRope : false;
            bool tieEmbeddings = val.valid ? val.config.tieEmbeddings : false;
            uint32_t tensors = val.valid ? val.tensorCount : static_cast<uint32_t>(meta.tensors.size());
            uint64_t weightBytes = val.valid ? val.totalWeightBytes : meta.totalWeightBytes;
            std::cout << "{"
                      << "\"status\":\"ok\","
                      << "\"valid\":true,"
                      << "\"version\":" << version << ","
                      << "\"architecture\":\"transformer\","
                      << "\"vocab_size\":" << vocabSize << ","
                      << "\"hidden_dim\":" << dim << ","
                      << "\"intermediate_dim\":" << interDim << ","
                      << "\"layers\":" << layers << ","
                      << "\"heads\":" << heads << ","
                      << "\"kv_heads\":" << kvHeads << ","
                      << "\"max_seq_len\":" << maxSeqLen << ","
                      << "\"rope_theta\":" << ropeTheta << ","
                      << "\"use_rope\":" << (useRope ? "true" : "false") << ","
                      << "\"tie_embeddings\":" << (tieEmbeddings ? "true" : "false") << ","
                      << "\"tensors\":" << tensors << ","
                      << "\"weight_bytes\":" << weightBytes
                      << "}" << std::endl;
            return 0;
        }

        if (arg == "--test-training-json") {
            int ret = NexaTrainer::runTrainingVerificationTests();
            std::cout << "{\"status\":\"" << (ret == 0 ? "ok" : "error") << "\",\"tests_passed\":" << (ret == 0 ? 4 : 0) << "}" << std::endl;
            return ret;
        }

        if (arg == "--train-nano-json") {
            uint32_t steps = 40;
            float lr = 3e-3f;
            if (argc > 2) steps = std::stoul(argv[2]);
            if (argc > 3) lr = std::stof(argv[3]);
            return runNanoTraining(steps, lr, true);
        }

        if (arg == "--generate-json") {
            if (argc < 4) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Usage: --generate-json <checkpoint> <prompt> [max_tokens] [temp] [top_k] [top_p] [rep_penalty] [seed] [nbt_path]\"}" << std::endl;
                return 1;
            }
            std::string ckpt = argv[2];
            std::string prompt = argv[3];
            uint32_t maxTokens = argc > 4 ? std::stoul(argv[4]) : 32;
            float temp = argc > 5 ? std::stof(argv[5]) : 0.7f;
            int32_t topK = argc > 6 ? std::stoi(argv[6]) : 40;
            float topP = argc > 7 ? std::stof(argv[7]) : 0.9f;
            float repPenalty = argc > 8 ? std::stof(argv[8]) : 1.15f;
            uint64_t seed = argc > 9 ? std::stoull(argv[9]) : 42;
            std::string nbtPath = argc > 10 ? argv[10] : "";
            return runGeneration(ckpt, prompt, maxTokens, temp, topK, topP, repPenalty, seed, true, nbtPath);
        }

        if (arg == "--validate-checkpoint-json") {
            if (argc < 3) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Missing checkpoint path\"}" << std::endl;
                return 1;
            }
            return runCheckpointValidation(argv[2], true);
        }

        if (arg == "--train-dataset-json") {
            if (argc < 3) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Usage: --train-dataset-json <dataset_file> [steps] [lr] [seq_len] [train_val_split] [checkpoint_path]\"}" << std::endl;
                return 1;
            }
            std::string ds = argv[2];
            uint32_t steps = argc > 3 ? std::stoul(argv[3]) : 50;
            float lr = argc > 4 ? std::stof(argv[4]) : 3e-3f;
            uint32_t seqLen = argc > 5 ? std::stoul(argv[5]) : 16;
            float split = argc > 6 ? std::stof(argv[6]) : 0.8f;
            std::string ckpt = argc > 7 ? argv[7] : "data/checkpoints/nexa_nano_trained.nexa";
            return runDatasetPipelineTraining(ds, steps, lr, seqLen, split, ckpt, true);
        }

        if (arg == "--test-generation-json") {
            int ret = NexaTrainer::runGenerationAndEvaluationTests();
            std::cout << "{\"status\":\"" << (ret == 0 ? "ok" : "error") << "\",\"tests_passed\":" << (ret == 0 ? 5 : 0) << "}" << std::endl;
            return ret;
        }

        if (arg == "--test-phase34-json") {
            int ret = NexaTrainer::runPhase34Tests();
            std::cout << "{\"status\":\"" << (ret == 0 ? "ok" : "error") << "\",\"tests_passed\":" << (ret == 0 ? 5 : 0) << "}" << std::endl;
            return ret;
        }

        if (arg == "--train-bpe-json") {
            if (argc < 4) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Usage: --train-bpe-json <corpus_file_or_dir> <output.nbt> [target_vocab] [min_freq]\"}" << std::endl;
                return 1;
            }
            std::string corpus = argv[2];
            std::string outNbt = argv[3];
            uint32_t targetVocab = argc > 4 ? std::stoul(argv[4]) : 280;
            uint32_t minFreq = argc > 5 ? std::stoul(argv[5]) : 2;
            return runBpeTrain(corpus, outNbt, targetVocab, minFreq, true);
        }

        if (arg == "--train-qa-json") {
            std::string dsDir = argc > 2 ? argv[2] : "data/datasets/qa";
            std::string ckpt = argc > 3 ? argv[3] : "data/checkpoints/nexa_nano_qa.nexa";
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_qa.nbt";
            uint32_t steps = argc > 5 ? std::stoul(argv[5]) : 60;
            uint32_t batchSize = argc > 6 ? std::stoul(argv[6]) : 2;
            uint32_t gradAccum = argc > 7 ? std::stoul(argv[7]) : 1;
            float lr = argc > 8 ? std::stof(argv[8]) : 3e-3f;
            uint32_t warmup = argc > 9 ? std::stoul(argv[9]) : 10;
            uint32_t seqLen = argc > 10 ? std::stoul(argv[10]) : 32;
            return runQaTraining(dsDir, ckpt, nbt, steps, batchSize, gradAccum, lr, warmup, seqLen, true);
        }

        if (arg == "--answer-json") {
            if (argc < 4) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Usage: --answer-json <checkpoint> <question> [nbt_path] [max_tokens] [temp] [seed]\"}" << std::endl;
                return 1;
            }
            std::string ckpt = argv[2];
            std::string question = argv[3];
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_qa.nbt";
            uint32_t maxTokens = argc > 5 ? std::stoul(argv[5]) : 32;
            float temp = argc > 6 ? std::stof(argv[6]) : 0.5f;
            uint64_t seed = argc > 7 ? std::stoull(argv[7]) : 42;
            return runQaAnswer(ckpt, question, nbt, maxTokens, temp, seed, true);
        }

        if (arg == "--benchmark-configs-json") {
            uint32_t vocab = argc > 2 ? std::stoul(argv[2]) : 768;
            return runBenchmarkConfigs(vocab, true);
        }

        if (arg == "--train-phase35-json") {
            std::string dsDir = argc > 2 ? argv[2] : "data/datasets/phase35/train_val";
            std::string bestCkpt = argc > 3 ? argv[3] : "data/checkpoints/nexa_phase35_best.nexa";
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_phase35.nbt";
            uint32_t steps = argc > 5 ? std::stoul(argv[5]) : 250;
            uint32_t dim = argc > 6 ? std::stoul(argv[6]) : 128;
            uint32_t layers = argc > 7 ? std::stoul(argv[7]) : 4;
            uint32_t heads = argc > 8 ? std::stoul(argv[8]) : 4;
            uint32_t vocab = argc > 9 ? std::stoul(argv[9]) : 768;
            uint32_t seqLen = argc > 10 ? std::stoul(argv[10]) : 64;
            uint32_t batch = argc > 11 ? std::stoul(argv[11]) : 2;
            uint32_t accum = argc > 12 ? std::stoul(argv[12]) : 2;
            float lr = argc > 13 ? std::stof(argv[13]) : 2.5e-3f;
            uint64_t seed = argc > 14 ? std::stoull(argv[14]) : 42;
            return runPhase35Training(dsDir, bestCkpt, nbt, steps, dim, layers, heads, vocab, seqLen, batch, accum, lr, seed, true);
        }

        if (arg == "--eval-unseen-json") {
            std::string ckpt = argc > 2 ? argv[2] : "data/checkpoints/nexa_phase35_best.nexa";
            std::string nbt = argc > 3 ? argv[3] : "data/tokenizers/nexa_bpe_phase35.nbt";
            std::string unseen = argc > 4 ? argv[4] : "data/datasets/phase35/unseen_test_qa.txt";
            std::string trainDir = argc > 5 ? argv[5] : "data/datasets/phase35/train_val";
            uint32_t maxTokens = argc > 6 ? std::stoul(argv[6]) : 48;
            float temp = argc > 7 ? std::stof(argv[7]) : 0.4f;
            uint64_t seed = argc > 8 ? std::stoull(argv[8]) : 42;
            return runUnseenEvaluation(ckpt, nbt, unseen, trainDir, maxTokens, temp, seed, true);
        }

        if (arg == "--test-phase35-json") {
            int ret = NexaTrainer::runPhase35Tests();
            std::cout << "{\"status\":\"" << (ret == 0 ? "ok" : "error") << "\",\"tests_passed\":" << (ret == 0 ? 5 : 0) << "}" << std::endl;
            return ret;
        }

        if (arg == "--test-phase36-json") {
            int ret = NexaTrainer::runPhase36Tests();
            std::cout << "{\"status\":\"" << (ret == 0 ? "ok" : "error") << "\",\"tests_passed\":" << (ret == 0 ? 6 : 0) << "}" << std::endl;
            return ret;
        }

        if (arg == "--audit-dataset-json") {
            if (argc < 3) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Usage: --audit-dataset-json <dataset_dir_or_manifest> [unseen_test_path] [nbt_path] [vocab] [require_verified_license 0|1]\"}" << std::endl;
                return 1;
            }
            std::string ds = argv[2];
            std::string unseen = argc > 3 ? argv[3] : "";
            std::string nbt = argc > 4 ? argv[4] : "";
            uint32_t vocab = argc > 5 ? std::stoul(argv[5]) : 768;
            bool reqLic = argc > 6 ? (std::stoi(argv[6]) != 0) : false;
            return runDatasetAudit(ds, unseen, nbt, vocab, reqLic, true);
        }

        if (arg == "--train-phase36-json") {
            if (argc < 3) {
                std::cerr << "{\"status\":\"error\",\"error\":\"Usage: --train-phase36-json <dataset_dir> [best_ckpt] [nbt] [unseen] [exp_id] [steps] [dim] [layers] [heads] [vocab] [seq_len] [batch] [accum] [lr] [patience] [seed] [require_lic] [use_rope] [tie_emb] [inter_dim] [resume_ckpt] [min_lr]\"}" << std::endl;
                return 1;
            }
            std::string ds = argv[2];
            std::string bestCkpt = argc > 3 ? argv[3] : "data/checkpoints/nexa_phase36_best.nexa";
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_phase36.nbt";
            std::string unseen = argc > 5 ? argv[5] : "";
            std::string expId = argc > 6 ? argv[6] : "";
            uint32_t steps = argc > 7 ? std::stoul(argv[7]) : 200;
            uint32_t dim = argc > 8 ? std::stoul(argv[8]) : 64;
            uint32_t layers = argc > 9 ? std::stoul(argv[9]) : 2;
            uint32_t heads = argc > 10 ? std::stoul(argv[10]) : 2;
            uint32_t vocab = argc > 11 ? std::stoul(argv[11]) : 768;
            uint32_t seqLen = argc > 12 ? std::stoul(argv[12]) : 64;
            uint32_t batch = argc > 13 ? std::stoul(argv[13]) : 2;
            uint32_t accum = argc > 14 ? std::stoul(argv[14]) : 2;
            float lr = argc > 15 ? std::stof(argv[15]) : 2.5e-3f;
            uint32_t patience = argc > 16 ? std::stoul(argv[16]) : 6;
            uint64_t seed = argc > 17 ? std::stoull(argv[17]) : 42;
            bool reqLic = argc > 18 ? (std::stoi(argv[18]) != 0) : false;
            bool useRope = argc > 19 ? (std::stoi(argv[19]) != 0) : false;
            bool tieEmb = argc > 20 ? (std::stoi(argv[20]) != 0) : false;
            uint32_t interDim = argc > 21 ? std::stoul(argv[21]) : 0;
            std::string resumeCkpt = argc > 22 ? argv[22] : "";
            float minLr = argc > 23 ? std::stof(argv[23]) : 1e-4f;
            return runPhase36Training(ds, bestCkpt, nbt, unseen, expId, steps, dim, layers, heads, vocab, seqLen, batch, accum, lr, patience, seed, reqLic, true, useRope, tieEmb, interDim, resumeCkpt, minLr);
        }

        if (arg == "--eval-unseen-v2-json") {
            std::string ckpt = argc > 2 ? argv[2] : "data/checkpoints/nexa_phase36_best.nexa";
            std::string nbt = argc > 3 ? argv[3] : "data/tokenizers/nexa_bpe_phase36.nbt";
            std::string unseen = argc > 4 ? argv[4] : "data/datasets/phase36/unseen_test_expanded.txt";
            std::string trainDir = argc > 5 ? argv[5] : "data/datasets/phase36/tier3_large/train_val";
            uint32_t maxTokens = argc > 6 ? std::stoul(argv[6]) : 48;
            float temp = argc > 7 ? std::stof(argv[7]) : 0.4f;
            uint64_t seed = argc > 8 ? std::stoull(argv[8]) : 42;
            return runUnseenEvaluation(ckpt, nbt, unseen, trainDir, maxTokens, temp, seed, true);
        }

        // Human-readable interactive commands
        printBanner();

        if (arg == "--test-tensor") {
            return runTensorTest();
        } else if (arg == "--tokenize" && argc > 2) {
            std::string nbtPath = argc > 3 ? argv[3] : "";
            std::unique_ptr<Tokenizer> tok;
            if (!nbtPath.empty()) {
                auto bpe = std::make_unique<NexaBpeTokenizer>();
                if (!bpe->load(nbtPath)) {
                    std::cerr << "Error: Failed to load BPE tokenizer: " << nbtPath << std::endl;
                    return 1;
                }
                tok = std::move(bpe);
            } else {
                tok = std::make_unique<NexaByteTokenizer>();
            }
            const bool addBos = nbtPath.empty();
            auto tokens = tok->encode(argv[2], addBos, false);
            std::cout << "Tokens: [";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i] << (i + 1 < tokens.size() ? ", " : "");
            }
            std::cout << "]" << std::endl;
            return 0;
        } else if (arg == "--inspect" && argc > 2) {
            auto val = NexaNanoModel::validateCheckpoint(argv[2]);
            auto meta = ModelLoader::inspectModel(argv[2]);
            if (!val.valid && !meta.isValid) {
                std::cerr << "Error: File is not a valid .nexa model format." << std::endl;
                return 1;
            }
            std::cout << "Model Architecture: Transformer Decoder" << std::endl;
            std::cout << "Format Version: " << (val.valid ? val.version : meta.header.version) << std::endl;
            std::cout << "Vocab Size: " << (val.valid ? val.config.vocabSize : meta.header.vocabSize) << std::endl;
            std::cout << "Hidden Dim: " << (val.valid ? val.config.dim : meta.header.dim) << std::endl;
            std::cout << "Intermediate Dim: " << (val.valid ? val.config.intermediateDim : meta.header.intermediateDim) << std::endl;
            std::cout << "Layers: " << (val.valid ? val.config.nLayers : meta.header.nLayers) << std::endl;
            std::cout << "Heads: " << (val.valid ? val.config.nHeads : meta.header.nHeads) << " (KV Heads: " << (meta.isValid ? meta.header.nKvHeads : val.config.nHeads) << ")" << std::endl;
            std::cout << "Max Seq Len: " << (val.valid ? val.config.maxSeqLen : meta.header.maxSeqLen) << std::endl;
            std::cout << "RoPE: " << (val.valid && val.config.useRope ? "enabled" : "disabled") << " | Tied Embeddings: " << (val.valid && val.config.tieEmbeddings ? "enabled" : "disabled") << std::endl;
            std::cout << "Tensors: " << (val.valid ? val.tensorCount : meta.tensors.size()) << std::endl;
            std::cout << "Weight Bytes: " << (val.valid ? val.totalWeightBytes : meta.totalWeightBytes) << std::endl;
            return 0;
        } else if (arg == "--test-training") {
            return NexaTrainer::runTrainingVerificationTests();
        } else if (arg == "--test-phase34") {
            return NexaTrainer::runPhase34Tests();
        } else if (arg == "--test-phase35") {
            return NexaTrainer::runPhase35Tests();
        } else if (arg == "--test-phase36") {
            return NexaTrainer::runPhase36Tests();
        } else if (arg == "--audit-dataset" && argc > 2) {
            std::string ds = argv[2];
            std::string unseen = argc > 3 ? argv[3] : "";
            std::string nbt = argc > 4 ? argv[4] : "";
            uint32_t vocab = argc > 5 ? std::stoul(argv[5]) : 768;
            bool reqLic = argc > 6 ? (std::stoi(argv[6]) != 0) : false;
            return runDatasetAudit(ds, unseen, nbt, vocab, reqLic, false);
        } else if (arg == "--train-phase36" && argc > 2) {
            std::string ds = argv[2];
            std::string bestCkpt = argc > 3 ? argv[3] : "data/checkpoints/nexa_phase36_best.nexa";
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_phase36.nbt";
            std::string unseen = argc > 5 ? argv[5] : "";
            std::string expId = argc > 6 ? argv[6] : "";
            uint32_t steps = argc > 7 ? std::stoul(argv[7]) : 200;
            uint32_t dim = argc > 8 ? std::stoul(argv[8]) : 64;
            uint32_t layers = argc > 9 ? std::stoul(argv[9]) : 2;
            uint32_t heads = argc > 10 ? std::stoul(argv[10]) : 2;
            uint32_t vocab = argc > 11 ? std::stoul(argv[11]) : 768;
            uint32_t seqLen = argc > 12 ? std::stoul(argv[12]) : 64;
            uint32_t batch = argc > 13 ? std::stoul(argv[13]) : 2;
            uint32_t accum = argc > 14 ? std::stoul(argv[14]) : 2;
            float lr = argc > 15 ? std::stof(argv[15]) : 2.5e-3f;
            uint32_t patience = argc > 16 ? std::stoul(argv[16]) : 6;
            uint64_t seed = argc > 17 ? std::stoull(argv[17]) : 42;
            bool reqLic = argc > 18 ? (std::stoi(argv[18]) != 0) : false;
            bool useRope = argc > 19 ? (std::stoi(argv[19]) != 0) : false;
            bool tieEmb = argc > 20 ? (std::stoi(argv[20]) != 0) : false;
            uint32_t interDim = argc > 21 ? std::stoul(argv[21]) : 0;
            std::string resumeCkpt = argc > 22 ? argv[22] : "";
            float minLr = argc > 23 ? std::stof(argv[23]) : 1e-4f;
            return runPhase36Training(ds, bestCkpt, nbt, unseen, expId, steps, dim, layers, heads, vocab, seqLen, batch, accum, lr, patience, seed, reqLic, false, useRope, tieEmb, interDim, resumeCkpt, minLr);
        } else if (arg == "--eval-unseen-v2") {
            std::string ckpt = argc > 2 ? argv[2] : "data/checkpoints/nexa_phase36_best.nexa";
            std::string nbt = argc > 3 ? argv[3] : "data/tokenizers/nexa_bpe_phase36.nbt";
            std::string unseen = argc > 4 ? argv[4] : "data/datasets/phase36/unseen_test_expanded.txt";
            std::string trainDir = argc > 5 ? argv[5] : "data/datasets/phase36/tier3_large/train_val";
            uint32_t maxTokens = argc > 6 ? std::stoul(argv[6]) : 48;
            float temp = argc > 7 ? std::stof(argv[7]) : 0.4f;
            uint64_t seed = argc > 8 ? std::stoull(argv[8]) : 42;
            return runUnseenEvaluation(ckpt, nbt, unseen, trainDir, maxTokens, temp, seed, false);
        } else if (arg == "--benchmark-configs") {
            uint32_t vocab = argc > 2 ? std::stoul(argv[2]) : 768;
            return runBenchmarkConfigs(vocab, false);
        } else if (arg == "--train-phase35") {
            std::string dsDir = argc > 2 ? argv[2] : "data/datasets/phase35/train_val";
            std::string bestCkpt = argc > 3 ? argv[3] : "data/checkpoints/nexa_phase35_best.nexa";
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_phase35.nbt";
            uint32_t steps = argc > 5 ? std::stoul(argv[5]) : 250;
            uint32_t dim = argc > 6 ? std::stoul(argv[6]) : 128;
            uint32_t layers = argc > 7 ? std::stoul(argv[7]) : 4;
            uint32_t heads = argc > 8 ? std::stoul(argv[8]) : 4;
            uint32_t vocab = argc > 9 ? std::stoul(argv[9]) : 768;
            uint32_t seqLen = argc > 10 ? std::stoul(argv[10]) : 64;
            uint32_t batch = argc > 11 ? std::stoul(argv[11]) : 2;
            uint32_t accum = argc > 12 ? std::stoul(argv[12]) : 2;
            float lr = argc > 13 ? std::stof(argv[13]) : 2.5e-3f;
            uint64_t seed = argc > 14 ? std::stoull(argv[14]) : 42;
            return runPhase35Training(dsDir, bestCkpt, nbt, steps, dim, layers, heads, vocab, seqLen, batch, accum, lr, seed, false);
        } else if (arg == "--eval-unseen") {
            std::string ckpt = argc > 2 ? argv[2] : "data/checkpoints/nexa_phase35_best.nexa";
            std::string nbt = argc > 3 ? argv[3] : "data/tokenizers/nexa_bpe_phase35.nbt";
            std::string unseen = argc > 4 ? argv[4] : "data/datasets/phase35/unseen_test_qa.txt";
            std::string trainDir = argc > 5 ? argv[5] : "data/datasets/phase35/train_val";
            uint32_t maxTokens = argc > 6 ? std::stoul(argv[6]) : 48;
            float temp = argc > 7 ? std::stof(argv[7]) : 0.4f;
            uint64_t seed = argc > 8 ? std::stoull(argv[8]) : 42;
            return runUnseenEvaluation(ckpt, nbt, unseen, trainDir, maxTokens, temp, seed, false);
        } else if (arg == "--train-nano") {
            uint32_t steps = 40;
            float lr = 3e-3f;
            if (argc > 2) steps = std::stoul(argv[2]);
            if (argc > 3) lr = std::stof(argv[3]);
            return runNanoTraining(steps, lr, false);
        } else if (arg == "--generate" && argc > 3) {
            std::string ckpt = argv[2];
            std::string prompt = argv[3];
            uint32_t maxTokens = argc > 4 ? std::stoul(argv[4]) : 32;
            float temp = argc > 5 ? std::stof(argv[5]) : 0.7f;
            int32_t topK = argc > 6 ? std::stoi(argv[6]) : 40;
            float topP = argc > 7 ? std::stof(argv[7]) : 0.9f;
            float repPenalty = argc > 8 ? std::stof(argv[8]) : 1.15f;
            uint64_t seed = argc > 9 ? std::stoull(argv[9]) : 42;
            std::string nbtPath = argc > 10 ? argv[10] : "";
            return runGeneration(ckpt, prompt, maxTokens, temp, topK, topP, repPenalty, seed, false, nbtPath);
        } else if (arg == "--validate-checkpoint" && argc > 2) {
            return runCheckpointValidation(argv[2], false);
        } else if (arg == "--train-dataset" && argc > 2) {
            std::string ds = argv[2];
            uint32_t steps = argc > 3 ? std::stoul(argv[3]) : 50;
            float lr = argc > 4 ? std::stof(argv[4]) : 3e-3f;
            uint32_t seqLen = argc > 5 ? std::stoul(argv[5]) : 16;
            float split = argc > 6 ? std::stof(argv[6]) : 0.8f;
            std::string ckpt = argc > 7 ? argv[7] : "data/checkpoints/nexa_nano_trained.nexa";
            return runDatasetPipelineTraining(ds, steps, lr, seqLen, split, ckpt, false);
        } else if (arg == "--test-generation") {
            return NexaTrainer::runGenerationAndEvaluationTests();
        } else if (arg == "--train-bpe" && argc > 3) {
            std::string corpus = argv[2];
            std::string outNbt = argv[3];
            uint32_t targetVocab = argc > 4 ? std::stoul(argv[4]) : 280;
            uint32_t minFreq = argc > 5 ? std::stoul(argv[5]) : 2;
            return runBpeTrain(corpus, outNbt, targetVocab, minFreq, false);
        } else if (arg == "--train-qa") {
            std::string dsDir = argc > 2 ? argv[2] : "data/datasets/qa";
            std::string ckpt = argc > 3 ? argv[3] : "data/checkpoints/nexa_nano_qa.nexa";
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_qa.nbt";
            uint32_t steps = argc > 5 ? std::stoul(argv[5]) : 60;
            uint32_t batchSize = argc > 6 ? std::stoul(argv[6]) : 2;
            uint32_t gradAccum = argc > 7 ? std::stoul(argv[7]) : 1;
            float lr = argc > 8 ? std::stof(argv[8]) : 3e-3f;
            uint32_t warmup = argc > 9 ? std::stoul(argv[9]) : 10;
            uint32_t seqLen = argc > 10 ? std::stoul(argv[10]) : 32;
            return runQaTraining(dsDir, ckpt, nbt, steps, batchSize, gradAccum, lr, warmup, seqLen, false);
        } else if (arg == "--answer" && argc > 3) {
            std::string ckpt = argv[2];
            std::string question = argv[3];
            std::string nbt = argc > 4 ? argv[4] : "data/tokenizers/nexa_bpe_qa.nbt";
            uint32_t maxTokens = argc > 5 ? std::stoul(argv[5]) : 32;
            float temp = argc > 6 ? std::stof(argv[6]) : 0.5f;
            uint64_t seed = argc > 7 ? std::stoull(argv[7]) : 42;
            return runQaAnswer(ckpt, question, nbt, maxTokens, temp, seed, false);
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            return 1;
        }
    }

    printBanner();
    std::cout << "Usage:" << std::endl;
    std::cout << "  nexa-runtime-cli --test-tensor                          (Run native tensor and NN math tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-training                        (Run autograd and AdamW unit tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-generation                      (Run Phase 3.3 generation & eval tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-phase34                         (Run Phase 3.4 BPE, dataset & scheduler tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-phase35                         (Run Phase 3.5 gradient, recovery & eval tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-phase36                         (Run Phase 3.6 data quality, masking & registry tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --audit-dataset <dir> [unseen] ...     (Audit dataset provenance, quality & contamination)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-phase36 <dir> [ckpt] [nbt] ... (Train Phase 3.6 model + save Experiment JSON)" << std::endl;
    std::cout << "  nexa-runtime-cli --eval-unseen-v2 [ckpt] [nbt] ...      (Multi-metric unseen evaluation)" << std::endl;
    std::cout << "  nexa-runtime-cli --benchmark-configs [vocab]            (Benchmark realistic model capacity configs)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-phase35 [dir] [ckpt] [nbt] ... (Train instruction-tuned Phase 3.5 model)" << std::endl;
    std::cout << "  nexa-runtime-cli --eval-unseen [ckpt] [nbt] [file] ...  (Evaluate unseen questions honestly)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-bpe <corpus> <out.nbt> [vocab] (Train trainable BPE tokenizer)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-qa [dir] [ckpt] [nbt] ...      (Train NEXA-1 Nano on QA dataset)" << std::endl;
    std::cout << "  nexa-runtime-cli --answer <ckpt> \"question\"             (Generate answer to natural language question)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-nano [steps] [lr]              (Train NEXA-1 Nano on synthetic data)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-dataset <file> [steps] [lr]    (Train on text dataset with train/val split)" << std::endl;
    std::cout << "  nexa-runtime-cli --generate <ckpt.nexa> \"prompt\"        (Autoregressive text generation)" << std::endl;
    std::cout << "  nexa-runtime-cli --validate-checkpoint <ckpt.nexa>      (Validate .nexa header and tensors)" << std::endl;
    std::cout << "  nexa-runtime-cli --tokenize \"text\"                      (Encode text to tokens)" << std::endl;
    std::cout << "  nexa-runtime-cli --inspect <file.nexa>                  (Inspect binary model header)" << std::endl;
    std::cout << "  nexa-runtime-cli --info-json                            (Machine-readable runtime telemetry)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-phase35-json                    (Machine-readable Phase 3.5 tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --test-phase36-json                    (Machine-readable Phase 3.6 tests)" << std::endl;
    std::cout << "  nexa-runtime-cli --audit-dataset-json <dir> ...         (Machine-readable dataset audit)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-phase36-json <dir> ...         (Machine-readable Phase 3.6 training)" << std::endl;
    std::cout << "  nexa-runtime-cli --eval-unseen-v2-json [ckpt] ...       (Machine-readable multi-metric unseen evaluation)" << std::endl;
    std::cout << "  nexa-runtime-cli --benchmark-configs-json [vocab]       (Machine-readable capacity benchmark)" << std::endl;
    std::cout << "  nexa-runtime-cli --train-phase35-json [dir] ...         (Machine-readable Phase 3.5 training)" << std::endl;
    std::cout << "  nexa-runtime-cli --eval-unseen-json [ckpt] ...          (Machine-readable unseen evaluation)" << std::endl;

    return 0;
}
