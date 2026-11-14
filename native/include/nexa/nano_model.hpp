/**
 * NEXA Native Training Engine - NEXA-1 Nano Model Header
 * Defines small decoder-only Transformer architecture, parameter tensors,
 * forward/backward graph execution, next-token prediction, autoregressive generation,
 * and checkpoint persistence.
 */

#ifndef NEXA_NANO_MODEL_HPP
#define NEXA_NANO_MODEL_HPP

#include "autograd.hpp"
#include "optimizer.hpp"
#include "tokenizer.hpp"
#include <vector>
#include <string>
#include <memory>
#include <random>

namespace nexa {

struct NanoConfig {
    uint32_t vocabSize = 256;      // Byte-level vocab for testing
    uint32_t dim = 64;             // Model hidden dimension
    uint32_t intermediateDim = 128;// FFN / MLP dimension
    uint32_t nLayers = 2;          // Number of Transformer layers
    uint32_t nHeads = 2;           // Multi-head attention heads
    uint32_t maxSeqLen = 32;       // Maximum context window
    float normEps = 1e-5f;
    float initStddev = 0.02f;
};

struct LayerParams {
    GradTensor attnNorm;
    GradTensor wQ;
    GradTensor wK;
    GradTensor wV;
    GradTensor wOut;
    GradTensor ffnNorm;
    GradTensor wGate;
    GradTensor wUp;
    GradTensor wDown;
};

struct LayerActivations {
    Tensor xNormAttn;
    Tensor invRmsAttn;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor attnScores;
    Tensor attnProbs;
    Tensor attnContext;
    Tensor attnOut;
    Tensor x1;
    Tensor xNormFfn;
    Tensor invRmsFfn;
    Tensor ffnGate;
    Tensor ffnUp;
    Tensor ffnSilu;
    Tensor ffnHidden;
    Tensor ffnOut;
    Tensor x2;
};

struct ModelActivations {
    std::vector<int32_t> tokens;
    Tensor embedOut;
    Tensor posOut;
    Tensor initialX;
    std::vector<LayerActivations> layers;
    Tensor finalNormOut;
    Tensor finalInvRms;
    Tensor logits;
    Tensor probs;
};

struct GenerationConfig {
    uint32_t maxNewTokens = 32;
    float temperature = 0.7f;        // 0.0 = greedy
    int32_t topK = 40;              // 0 = disabled
    float topP = 0.9f;              // 1.0 = disabled
    float repetitionPenalty = 1.15f; // 1.0 = no penalty
    int32_t stopToken = -1;         // default -1 (or EOS = 2)
    std::vector<std::string> stopSequences; // e.g. {"\nQ:", "\nQuestion:", "<|end_of_text|>"}
    uint64_t seed = 42;             // seed for reproducible sampling
};

struct GenerationResult {
    std::string prompt;
    std::string generatedText;
    std::string fullText;
    std::vector<int32_t> promptTokens;
    std::vector<int32_t> generatedTokens;
    uint32_t tokensGenerated = 0;
    double durationMs = 0.0;
    double tokensPerSecond = 0.0;
    std::string finishReason; // "length", "stop_token"
    bool success = true;
    std::string errorMessage;
};

struct CheckpointValidationResult {
    bool valid = false;
    uint32_t version = 0;
    uint32_t archType = 0;
    NanoConfig config;
    uint64_t step = 0;
    float loss = 0.0f;
    bool hasOptimizerState = false;
    uint32_t tensorCount = 0;
    uint64_t totalWeightBytes = 0;
    std::string error;
};

struct ModelMemoryEstimate {
    int64_t totalParameters = 0;
    uint64_t weightsBytes = 0;
    uint64_t gradientsBytes = 0;
    uint64_t optimizerBytes = 0;
    uint64_t activationsBytes = 0;
    uint64_t totalTrainingBytes = 0;
    uint64_t totalInferenceBytes = 0;
};

struct ArchitecturePrepSpec {
    bool rmsNormActive = true;              // Active in forward & backward autograd
    bool ropePrepared = true;               // Planned future positional encoding
    bool ropeActiveInTraining = false;      // Not yet connected to autograd graph
    bool tiedEmbeddingsPrepared = true;     // Planned future weight-tying support
    bool tiedEmbeddingsActiveInTraining = false;
    float plannedDropoutRate = 0.0f;        // Planned regularization rate
    bool dropoutActiveInTraining = false;
    uint32_t plannedLongContextLen = 512;   // Target context window capacity check
    uint32_t checkpointFormatVersion = 2;   // Preserves v1/v2 .nexa compatibility
};

class NexaNanoModel {
public:
    NanoConfig config;
    GradTensor wEmbed;     // [vocabSize, dim]
    GradTensor wPos;       // [maxSeqLen, dim]
    std::vector<LayerParams> layers;
    GradTensor finalNorm;  // [dim]
    GradTensor wHead;      // [dim, vocabSize]

    std::vector<GradTensor*> allParams;

    NexaNanoModel() = default;
    explicit NexaNanoModel(const NanoConfig& cfg, uint64_t initSeed = 1337);

    void initWeights(uint64_t seed);
    void zeroGrad();

    std::vector<GradTensor*>& getParameters() { return allParams; }
    const std::vector<GradTensor*>& getParameters() const { return allParams; }

    /**
     * Inference Forward Pass: Computes output logits [seqLen, vocabSize]
     */
    void forward(const std::vector<int32_t>& inputTokens, Tensor& logitsOut);

    /**
     * Training Forward + Loss + Backward Pass:
     * Computes next-token prediction cross-entropy loss, and backpropagates gradients
     * into all model parameters. Supports ignoreIndex masking for instruction tuning.
     */
    float forwardAndLoss(
        const std::vector<int32_t>& inputTokens,
        const std::vector<int32_t>& targetTokens,
        ModelActivations& act,
        int32_t ignoreIndex = -1
    );

    /**
     * Samples next token with Temperature, Top-K, Top-P, and Repetition Penalty.
     */
    int32_t sampleNextToken(
        const std::vector<float>& logits,
        const std::vector<int32_t>& contextTokens,
        const GenerationConfig& cfg,
        std::mt19937_64& rng
    );

    /**
     * Generates greedy next token from prompt.
     */
    int32_t predictNextToken(const std::vector<int32_t>& inputTokens);

    /**
     * Full Autoregressive Text Generation:
     * Tokenizes prompt, runs token-by-token loop with temperature/top-k/top-p sampling,
     * repetition penalty, and decodes back to UTF-8 text.
     */
    GenerationResult generateText(
        const std::string& prompt,
        Tokenizer& tokenizer,
        const GenerationConfig& genCfg
    );

    /**
     * Checkpoint persistence in extended .nexa binary format.
     */
    bool saveCheckpoint(
        const std::string& filePath,
        uint64_t step = 0,
        float loss = 0.0f,
        const AdamW* opt = nullptr
    ) const;

    /**
     * Validates checkpoint binary header, tensor descriptors, and architecture compatibility.
     */
    static CheckpointValidationResult validateCheckpoint(const std::string& filePath);

    /**
     * Strict checkpoint loader with dimension and integrity validation.
     */
    bool loadCheckpoint(
        const std::string& filePath,
        uint64_t* outStep = nullptr,
        float* outLoss = nullptr,
        AdamW* opt = nullptr,
        std::string* outError = nullptr
    );

    int64_t totalParameters() const;
    static ModelMemoryEstimate estimateMemory(const NanoConfig& cfg);
    static ModelMemoryEstimate estimateMemoryWithOptions(
        const NanoConfig& cfg,
        bool simulateTiedEmbeddings = false,
        bool simulateRopeNoWPos = false,
        uint32_t overrideMaxSeqLen = 0
    );
    static ArchitecturePrepSpec getArchitecturePrepSpec();
};

} // namespace nexa

#endif // NEXA_NANO_MODEL_HPP
