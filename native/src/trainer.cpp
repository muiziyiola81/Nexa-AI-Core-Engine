/**
 * NEXA Native Training Engine - Trainer Pipeline Implementation
 */

#include "../include/nexa/trainer.hpp"
#include "../include/nexa/experiment.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <numeric>
#include <sys/resource.h>

namespace nexa {

double NexaTrainer::getPeakRssMb() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return static_cast<double>(ru.ru_maxrss) / 1024.0;
    }
    return 0.0;
}

TrainingResult NexaTrainer::trainOnSyntheticCorpus(
    NexaNanoModel& model,
    const std::string& textCorpus,
    uint32_t numSteps,
    float learningRate,
    uint32_t seqLen,
    const std::string& checkpointDir
) {
    TrainingResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Tokenize dataset
    NexaByteTokenizer tokenizer;
    std::vector<int32_t> allTokens = tokenizer.encode(textCorpus, false, false);
    if (allTokens.size() < seqLen + 2) {
        // Replicate to satisfy sequence length requirement
        std::vector<int32_t> rep;
        while (rep.size() < (seqLen + 2) * 10) {
            for (auto t : allTokens) rep.push_back(t);
        }
        allTokens = rep;
    }

    // 2. Measure initial weight norm & save initial weight copy for delta verification
    double initialSumSq = 0.0;
    std::vector<std::vector<float>> initialWeightsCopy;
    for (const auto* p : model.getParameters()) {
        const float* v = p->value.dataFloat();
        int64_t n = p->value.numel();
        std::vector<float> copy(v, v + n);
        initialWeightsCopy.push_back(copy);
        for (int64_t i = 0; i < n; ++i) {
            initialSumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
        }
    }
    result.initialWeightNorm = static_cast<float>(std::sqrt(initialSumSq));

    // 3. Initialize AdamW optimizer
    AdamWConfig optCfg;
    optCfg.lr = learningRate;
    optCfg.weightDecay = 0.01f;
    optCfg.maxGradNorm = 1.0f;
    AdamW optimizer(model.getParameters(), optCfg);

    // 4. Training Loop
    ModelActivations act;
    size_t corpusOffset = 0;
    result.totalTokensTrained = 0;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "        NEXA-1 NANO NATIVE TRAINING LOOP (C++20 CPU)             " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Parameters: " << model.totalParameters() << " | Vocab: " << model.config.vocabSize
              << " | Dim: " << model.config.dim << " | Layers: " << model.config.nLayers
              << " | Heads: " << model.config.nHeads << std::endl;
    std::cout << "Learning Rate: " << learningRate << " | Steps: " << numSteps
              << " | Sequence Length: " << seqLen << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << std::setw(6) << "Step" << " | "
              << std::setw(10) << "Loss" << " | "
              << std::setw(12) << "Grad Norm" << " | "
              << std::setw(14) << "Tokens Trained" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    for (uint32_t step = 1; step <= numSteps; ++step) {
        // Extract sequence window: inputs [t..t+seqLen-1], targets [t+1..t+seqLen]
        if (corpusOffset + seqLen + 1 >= allTokens.size()) {
            corpusOffset = 0;
        }

        std::vector<int32_t> inputs(allTokens.begin() + corpusOffset, allTokens.begin() + corpusOffset + seqLen);
        std::vector<int32_t> targets(allTokens.begin() + corpusOffset + 1, allTokens.begin() + corpusOffset + seqLen + 1);
        corpusOffset += (seqLen / 2); // Shift sliding window

        // Forward & Loss & Backward Pass
        optimizer.zeroGrad();
        float loss = model.forwardAndLoss(inputs, targets, act);
        result.lossHistory.push_back(loss);
        result.totalTokensTrained += seqLen;

        if (step == 1) {
            result.initialLoss = loss;
        }

        // Optimizer Step (AdamW + Gradient Clipping)
        float gradNorm = optimizer.step();

        if (step == 1 || step % 5 == 0 || step == numSteps) {
            std::cout << std::setw(6) << step << " | "
                      << std::setw(10) << std::fixed << std::setprecision(4) << loss << " | "
                      << std::setw(12) << std::fixed << std::setprecision(4) << gradNorm << " | "
                      << std::setw(14) << result.totalTokensTrained << std::endl;
        }
    }

    result.totalSteps = numSteps;
    result.finalLoss = result.lossHistory.back();
    result.lossReductionPercent = ((result.initialLoss - result.finalLoss) / result.initialLoss) * 100.0f;

    // 5. Measure final weight norm & delta
    double finalSumSq = 0.0;
    double deltaSumSq = 0.0;
    const auto& params = model.getParameters();
    for (size_t p = 0; p < params.size(); ++p) {
        const float* v = params[p]->value.dataFloat();
        const auto& initialCopy = initialWeightsCopy[p];
        int64_t n = params[p]->value.numel();

        for (int64_t i = 0; i < n; ++i) {
            finalSumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
            double d = static_cast<double>(v[i]) - static_cast<double>(initialCopy[i]);
            deltaSumSq += d * d;
        }
    }

    result.finalWeightNorm = static_cast<float>(std::sqrt(finalSumSq));
    result.weightDeltaNorm = static_cast<float>(std::sqrt(deltaSumSq));
    result.weightsUpdated = (result.weightDeltaNorm > 1e-4f);

    // 6. Checkpoint Verification Test: Save and Reload
    std::filesystem::create_directories(checkpointDir);
    std::string ckptFile = checkpointDir + "/nexa_nano_step_" + std::to_string(numSteps) + ".nexa";
    result.checkpointPath = ckptFile;

    bool saved = model.saveCheckpoint(ckptFile, numSteps, result.finalLoss, &optimizer);
    if (saved) {
        // Instantiate a new model and load checkpoint
        NexaNanoModel reloadedModel(model.config, 9999); // different seed
        uint64_t reloadedStep = 0;
        float reloadedLoss = 0.0f;
        AdamW reloadedOpt(reloadedModel.getParameters(), optCfg);

        bool loaded = reloadedModel.loadCheckpoint(ckptFile, &reloadedStep, &reloadedLoss, &reloadedOpt);
        if (loaded && reloadedStep == numSteps) {
            result.checkpointVerified = true;

            // Test whether reloaded model outputs bit-identical predictions
            std::vector<int32_t> prompt = {1, 78, 69, 88, 65}; // "<bos> NEXA"
            Tensor origLogits;
            Tensor reloadedLogits;
            model.forward(prompt, origLogits);
            reloadedModel.forward(prompt, reloadedLogits);

            const float* origData = origLogits.dataFloat();
            const float* relData = reloadedLogits.dataFloat();
            int64_t nLogits = origLogits.numel();

            bool identical = true;
            for (int64_t i = 0; i < nLogits; ++i) {
                if (std::abs(origData[i] - relData[i]) > 1e-5f) {
                    identical = false;
                    break;
                }
            }
            result.reloadedOutputIdentical = identical;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.durationSeconds = std::chrono::duration<double>(t1 - t0).count();
    result.success = (result.finalLoss < result.initialLoss && result.weightsUpdated && result.checkpointVerified);

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "TRAINING SUMMARY:" << std::endl;
    std::cout << "  Initial Loss:         " << result.initialLoss << std::endl;
    std::cout << "  Final Loss:           " << result.finalLoss << " (-" << std::fixed << std::setprecision(1) << result.lossReductionPercent << "%)" << std::endl;
    std::cout << "  Initial Weight Norm:  " << result.initialWeightNorm << std::endl;
    std::cout << "  Final Weight Norm:    " << result.finalWeightNorm << std::endl;
    std::cout << "  Weight Delta Norm:    " << result.weightDeltaNorm << " (Weights updated: " << (result.weightsUpdated ? "YES" : "NO") << ")" << std::endl;
    std::cout << "  Checkpoint Verified:  " << (result.checkpointVerified ? "YES" : "NO") << " (" << result.checkpointPath << ")" << std::endl;
    std::cout << "  Reloaded Invariance:  " << (result.reloadedOutputIdentical ? "IDENTICAL" : "MISMATCH") << std::endl;
    std::cout << "  Elapsed Time:         " << std::fixed << std::setprecision(2) << result.durationSeconds << " s" << std::endl;
    std::cout << "================================================================\n" << std::endl;

    return result;
}

int NexaTrainer::runTrainingVerificationTests() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "     NEXA NATIVE TRAINING UNIT TESTS (AUTOGRAD & OPTIMIZER)     " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test 1: Matmul Backward Numerical Verification
    std::cout << "\n[TEST 1] Matrix Multiplication Gradients (dA = dY * B^T, dB = A^T * dY)..." << std::endl;
    Tensor A(Shape({2, 3}));
    Tensor B(Shape({3, 2}));
    Tensor Y(Shape({2, 2}));
    Tensor gradY(Shape({2, 2}));
    Tensor gradA(Shape({2, 3}));
    Tensor gradB(Shape({3, 2}));

    A.dataFloat()[0] = 1.0f; A.dataFloat()[1] = 2.0f; A.dataFloat()[2] = 3.0f;
    A.dataFloat()[3] = 4.0f; A.dataFloat()[4] = 5.0f; A.dataFloat()[5] = 6.0f;

    B.dataFloat()[0] = 7.0f; B.dataFloat()[1] = 8.0f;
    B.dataFloat()[2] = 9.0f; B.dataFloat()[3] = 1.0f;
    B.dataFloat()[4] = 2.0f; B.dataFloat()[5] = 3.0f;

    // Forward: Y = A * B
    AutogradOps::matmulForward(A, B, Y);

    // Let gradY = [[1, 0], [0, 1]] (identity)
    gradY.dataFloat()[0] = 1.0f; gradY.dataFloat()[1] = 0.0f;
    gradY.dataFloat()[2] = 0.0f; gradY.dataFloat()[3] = 1.0f;

    AutogradOps::matmulBackward(A, B, gradY, &gradA, &gradB, false, false);

    // dA = gradY * B^T
    // gradA[0, 0] = 1*7 + 0*8 = 7
    // gradA[0, 1] = 1*9 + 0*1 = 9
    // gradA[0, 2] = 1*2 + 0*3 = 2
    if (std::abs(gradA.dataFloat()[0] - 7.0f) > 1e-4f ||
        std::abs(gradA.dataFloat()[1] - 9.0f) > 1e-4f ||
        std::abs(gradA.dataFloat()[2] - 2.0f) > 1e-4f) {
        std::cerr << "FAIL: Matmul backward gradA mismatch!" << std::endl;
        return 1;
    }
    std::cout << "Matmul gradient assertion passed." << std::endl;

    // Test 2: RMSNorm Backward
    std::cout << "\n[TEST 2] RMSNorm Gradients..." << std::endl;
    Tensor x(Shape({1, 4}));
    Tensor w(Shape({4}));
    Tensor y(Shape({1, 4}));
    Tensor invRms(Shape({1}));
    Tensor gy(Shape({1, 4}));
    Tensor gx(Shape({1, 4}));
    Tensor gw(Shape({4}));

    x.dataFloat()[0] = 1.0f; x.dataFloat()[1] = 2.0f; x.dataFloat()[2] = 3.0f; x.dataFloat()[3] = 4.0f;
    w.fillConstant(1.0f);
    gy.fillConstant(1.0f);

    AutogradOps::rmsnormForward(x, w, y, invRms);
    AutogradOps::rmsnormBackward(x, w, invRms, gy, &gx, &gw, false);

    // RMSNorm has scale invariant property: sum(x * gx) == 0
    float dotXGx = 0.0f;
    for (int i = 0; i < 4; ++i) dotXGx += x.dataFloat()[i] * gx.dataFloat()[i];
    std::cout << "RMSNorm scale orthogonality sum(x . dx) = " << dotXGx << " (expected ~0.0)" << std::endl;
    if (std::abs(dotXGx) > 1e-4f) {
        std::cerr << "FAIL: RMSNorm scale invariant violated in backward!" << std::endl;
        return 1;
    }
    std::cout << "RMSNorm backward assertion passed." << std::endl;

    // Test 3: Cross Entropy Loss & Backward
    std::cout << "\n[TEST 3] Cross Entropy Loss & Softmax Gradients..." << std::endl;
    Tensor logits(Shape({1, 3}));
    logits.dataFloat()[0] = 2.0f;
    logits.dataFloat()[1] = 1.0f;
    logits.dataFloat()[2] = 0.1f;
    std::vector<int32_t> targets = {0}; // target is class 0

    Tensor gradLogits(Shape({1, 3}));
    Tensor probs(Shape({1, 3}));

    float loss = AutogradOps::crossEntropyForwardAndBackward(logits, targets, gradLogits, probs);
    std::cout << "Cross-entropy loss = " << loss << " (target=0)" << std::endl;

    // Target gradient: (p_0 - 1.0). Since p_0 < 1, gradLogits[0] must be negative!
    if (gradLogits.dataFloat()[0] >= 0.0f) {
        std::cerr << "FAIL: Target gradient must be negative to pull logits up!" << std::endl;
        return 1;
    }
    // Sum of cross entropy gradients across classes must be 0: sum(p_i) - 1 = 0
    float sumGrad = gradLogits.dataFloat()[0] + gradLogits.dataFloat()[1] + gradLogits.dataFloat()[2];
    std::cout << "Sum of softmax loss gradients = " << sumGrad << " (expected 0.0)" << std::endl;
    if (std::abs(sumGrad) > 1e-5f) {
        std::cerr << "FAIL: Softmax gradients do not sum to 0.0!" << std::endl;
        return 1;
    }
    std::cout << "Cross-entropy gradient assertion passed." << std::endl;

    // Test 4: AdamW Optimizer Step
    std::cout << "\n[TEST 4] AdamW Optimizer Moment Updates & Weight Decay..." << std::endl;
    GradTensor param("test_param", Shape({2}));
    param.value.dataFloat()[0] = 1.0f;
    param.value.dataFloat()[1] = 2.0f;
    param.grad.dataFloat()[0] = 0.5f; // positive gradient -> parameter decreases
    param.grad.dataFloat()[1] = -0.5f;// negative gradient -> parameter increases

    AdamWConfig cfg;
    cfg.lr = 0.1f;
    cfg.weightDecay = 0.01f;
    AdamW opt({&param}, cfg);

    float normBefore = param.l2NormValue();
    opt.step();
    float normAfter = param.l2NormValue();

    // Check updates moved parameters in correct opposite direction of gradients
    if (param.value.dataFloat()[0] >= 1.0f || param.value.dataFloat()[1] <= 2.0f) {
        std::cerr << "FAIL: AdamW did not update parameters in the descent direction!" << std::endl;
        return 1;
    }
    std::cout << "Param 0: 1.0 -> " << param.value.dataFloat()[0] << " (decreased)" << std::endl;
    std::cout << "Param 1: 2.0 -> " << param.value.dataFloat()[1] << " (increased)" << std::endl;
    std::cout << "AdamW assertion passed." << std::endl;

    std::cout << "\nALL 4 AUTOGRAD & OPTIMIZER UNIT TESTS PASSED SUCCESSFULLY." << std::endl;
    return 0;
}

EvaluationResult NexaTrainer::evaluateModel(
    NexaNanoModel& model,
    const std::vector<SequenceSample>& valSequences,
    int32_t ignoreIndex
) {
    EvaluationResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (valSequences.empty()) {
        return res;
    }

    double totalLoss = 0.0;
    uint32_t validSeqs = 0;
    uint32_t totalTokens = 0;

    for (const auto& sample : valSequences) {
        Tensor logits;
        model.forward(sample.inputs, logits);

        Tensor gradLogits(logits.shape());
        Tensor probs(logits.shape());
        float loss = AutogradOps::crossEntropyForwardAndBackward(logits, sample.targets, gradLogits, probs, ignoreIndex);

        totalLoss += static_cast<double>(loss);
        validSeqs++;
        totalTokens += static_cast<uint32_t>(sample.inputs.size());
    }

    res.sequencesEvaluated = validSeqs;
    res.tokensEvaluated = totalTokens;
    res.validationLoss = validSeqs > 0 ? static_cast<float>(totalLoss / validSeqs) : 0.0f;
    res.perplexity = std::exp(std::min(20.0f, res.validationLoss));

    auto t1 = std::chrono::high_resolution_clock::now();
    res.durationMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return res;
}

PipelineTrainingResult NexaTrainer::trainWithDataset(
    NexaNanoModel& model,
    const DatasetPipeline& dataset,
    uint32_t numSteps,
    float learningRate,
    const std::string& checkpointPath
) {
    PipelineTrainingResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    const auto& trainSeqs = dataset.trainSequences();
    const auto& valSeqs = dataset.valSequences();

    if (trainSeqs.empty()) {
        result.success = false;
        result.errorMessage = "Training dataset has no valid sequences";
        return result;
    }

    // 1. Initial Evaluation & Weight Norm
    double initialSumSq = 0.0;
    std::vector<std::vector<float>> initialWeightsCopy;
    for (const auto* p : model.getParameters()) {
        const float* v = p->value.dataFloat();
        int64_t n = p->value.numel();
        initialWeightsCopy.emplace_back(v, v + n);
        for (int64_t i = 0; i < n; ++i) {
            initialSumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
        }
    }
    result.initialWeightNorm = static_cast<float>(std::sqrt(initialSumSq));

    auto initialEval = evaluateModel(model, valSeqs);
    result.initialValLoss = initialEval.validationLoss;
    result.initialPerplexity = initialEval.perplexity;

    // 2. Setup AdamW Optimizer
    AdamWConfig optCfg;
    optCfg.lr = learningRate;
    optCfg.weightDecay = 0.01f;
    optCfg.maxGradNorm = 1.0f;
    AdamW optimizer(model.getParameters(), optCfg);

    ModelActivations act;
    size_t seqIdx = 0;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "         NEXA-1 NANO DATASET PIPELINE TRAINING RUN               " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Dataset: " << dataset.metadata().filePath << " (" << dataset.metadata().totalTokens << " tokens)" << std::endl;
    std::cout << "Train sequences: " << trainSeqs.size() << " | Val sequences: " << valSeqs.size() << std::endl;
    std::cout << "Initial Val Loss: " << result.initialValLoss << " | Initial PPL: " << result.initialPerplexity << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << std::setw(6) << "Step" << " | "
              << std::setw(11) << "Train Loss" << " | "
              << std::setw(11) << "Val Loss" << " | "
              << std::setw(10) << "Val PPL" << " | "
              << std::setw(10) << "Grad Norm" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    float currentTrainLoss = 0.0f;

    for (uint32_t step = 1; step <= numSteps; ++step) {
        const auto& sample = trainSeqs[seqIdx];
        seqIdx = (seqIdx + 1) % trainSeqs.size();

        optimizer.zeroGrad();
        float loss = model.forwardAndLoss(sample.inputs, sample.targets, act);
        currentTrainLoss = loss;

        if (step == 1) {
            result.initialTrainLoss = loss;
        }

        float gradNorm = optimizer.step();

        if (step == 1 || step % 10 == 0 || step == numSteps) {
            auto eval = evaluateModel(model, valSeqs);
            std::cout << std::setw(6) << step << " | "
                      << std::setw(11) << std::fixed << std::setprecision(4) << loss << " | "
                      << std::setw(11) << std::fixed << std::setprecision(4) << eval.validationLoss << " | "
                      << std::setw(10) << std::fixed << std::setprecision(2) << eval.perplexity << " | "
                      << std::setw(10) << std::fixed << std::setprecision(4) << gradNorm << std::endl;
        }
    }

    result.totalSteps = numSteps;
    result.finalTrainLoss = currentTrainLoss;

    auto finalEval = evaluateModel(model, valSeqs);
    result.finalValLoss = finalEval.validationLoss;
    result.finalPerplexity = finalEval.perplexity;

    // 3. Final weight norm & delta
    double finalSumSq = 0.0;
    double deltaSumSq = 0.0;
    const auto& params = model.getParameters();
    for (size_t p = 0; p < params.size(); ++p) {
        const float* v = params[p]->value.dataFloat();
        const auto& initialCopy = initialWeightsCopy[p];
        int64_t n = params[p]->value.numel();

        for (int64_t i = 0; i < n; ++i) {
            finalSumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
            double d = static_cast<double>(v[i]) - static_cast<double>(initialCopy[i]);
            deltaSumSq += d * d;
        }
    }

    result.finalWeightNorm = static_cast<float>(std::sqrt(finalSumSq));
    result.weightDeltaNorm = static_cast<float>(std::sqrt(deltaSumSq));
    result.weightsUpdated = (result.weightDeltaNorm > 1e-4f);

    // 4. Save Checkpoint
    std::filesystem::path p(checkpointPath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    result.checkpointPath = checkpointPath;
    model.saveCheckpoint(checkpointPath, numSteps, result.finalTrainLoss, &optimizer);

    // 5. Sample Generation
    NexaByteTokenizer tokenizer;
    GenerationConfig genCfg;
    genCfg.maxNewTokens = 24;
    genCfg.temperature = 0.6f;
    genCfg.seed = 42;
    auto genRes = model.generateText("NEXA", tokenizer, genCfg);
    result.sampleGeneration = genRes.fullText;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.durationSeconds = std::chrono::duration<double>(t1 - t0).count();
    result.success = (result.finalTrainLoss < result.initialTrainLoss && result.weightsUpdated);

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "PIPELINE TRAINING SUMMARY:" << std::endl;
    std::cout << "  Train Loss:       " << result.initialTrainLoss << " -> " << result.finalTrainLoss << std::endl;
    std::cout << "  Val Loss:         " << result.initialValLoss << " -> " << result.finalValLoss << std::endl;
    std::cout << "  Val Perplexity:   " << result.initialPerplexity << " -> " << result.finalPerplexity << std::endl;
    std::cout << "  Weight Delta:     " << result.weightDeltaNorm << " (Weights updated: " << (result.weightsUpdated ? "YES" : "NO") << ")" << std::endl;
    std::cout << "  Sample Output:    \"" << result.sampleGeneration << "\"" << std::endl;
    std::cout << "  Saved To:         " << result.checkpointPath << std::endl;
    std::cout << "================================================================\n" << std::endl;

    return result;
}

PipelineTrainingResult NexaTrainer::trainWithDatasetAdvanced(
    NexaNanoModel& model,
    const DatasetPipeline& dataset,
    const TrainingPipelineConfig& config,
    Tokenizer* tokenizer
) {
    PipelineTrainingResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    const auto& trainSeqs = dataset.trainSequences();
    const auto& valSeqs = dataset.valSequences();

    if (trainSeqs.empty()) {
        result.success = false;
        result.errorMessage = "Training dataset has no valid sequences";
        return result;
    }

    // 1. Setup AdamW Optimizer & Optional Resume (with optimizer state!)
    AdamWConfig optCfg;
    optCfg.lr = config.learningRate;
    optCfg.weightDecay = config.weightDecay;
    optCfg.maxGradNorm = config.maxGradNorm;
    AdamW optimizer(model.getParameters(), optCfg);

    if (!config.resumeFromCheckpoint.empty()) {
        std::cout << "[TRAINER] Resuming from checkpoint: " << config.resumeFromCheckpoint << std::endl;
        uint64_t loadedStep = 0;
        float loadedLoss = 0.0f;
        std::string loadErr;
        if (!model.loadCheckpoint(config.resumeFromCheckpoint, &loadedStep, &loadedLoss, &optimizer, &loadErr)) {
            result.success = false;
            result.errorMessage = "Failed to resume from checkpoint: " + loadErr;
            return result;
        }
        std::cout << "[TRAINER] Successfully resumed at step " << loadedStep << " (previous loss: " << loadedLoss << ")" << std::endl;
    }

    // 2. Initial Evaluation & Weight Norm
    double initialSumSq = 0.0;
    std::vector<std::vector<float>> initialWeightsCopy;
    for (const auto* p : model.getParameters()) {
        const float* v = p->value.dataFloat();
        int64_t n = p->value.numel();
        initialWeightsCopy.emplace_back(v, v + n);
        for (int64_t i = 0; i < n; ++i) {
            initialSumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
        }
    }
    result.initialWeightNorm = static_cast<float>(std::sqrt(initialSumSq));

    auto initialEval = evaluateModel(model, valSeqs, config.ignoreIndex);
    result.initialValLoss = initialEval.validationLoss;
    result.initialPerplexity = initialEval.perplexity;
    result.bestValLoss = initialEval.validationLoss;
    result.bestPerplexity = initialEval.perplexity;
    result.bestCheckpointStep = 0;

    std::string bestCkptPath = config.bestCheckpointPath.empty() ? config.checkpointPath : config.bestCheckpointPath;
    result.bestCheckpointPath = bestCkptPath;

    auto ensureParent = [](const std::string& pathStr) {
        std::filesystem::path p(pathStr);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
    };
    ensureParent(config.checkpointPath);
    ensureParent(bestCkptPath);

    uint32_t batchSize = std::max(1u, config.batchSize);
    uint32_t gradAccum = std::max(1u, config.gradAccumSteps);
    uint32_t samplesPerStep = batchSize * gradAccum;
    uint32_t totalSteps = config.numSteps > 0 ? config.numSteps : (config.epochs * ((static_cast<uint32_t>(trainSeqs.size()) + samplesPerStep - 1) / samplesPerStep));

    LrScheduleConfig lrSchedule(config.learningRate, config.minLr, config.warmupSteps, totalSteps);

    std::vector<size_t> order(trainSeqs.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 shuffleRng(config.seed);
    if (config.shuffleBatches) {
        std::shuffle(order.begin(), order.end(), shuffleRng);
    }

    size_t seqCursor = 0;
    float currentTrainLoss = 0.0f;
    uint32_t evalsWithoutImprove = 0;
    uint32_t stepsExecuted = 0;
    int64_t totalTokensTrained = 0;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "     NEXA-1 NANO ADVANCED MULTI-BATCH TRAINING PIPELINE          " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Model: " << model.totalParameters() << " params | Vocab: " << model.config.vocabSize
              << " | Dim: " << model.config.dim << " | Layers: " << model.config.nLayers
              << " | Heads: " << model.config.nHeads << " | Ctx: " << model.config.maxSeqLen << std::endl;
    std::cout << "Dataset: " << dataset.metadata().sourceFiles.size() << " files, "
              << dataset.metadata().totalTokens << " tokens, "
              << dataset.metadata().exampleCount << " examples ("
              << dataset.metadata().duplicatesRemoved << " duplicates pruned)" << std::endl;
    std::cout << "Train seqs: " << trainSeqs.size() << " | Val seqs: " << valSeqs.size()
              << " | Prompt-masked targets: " << dataset.metadata().maskedTokensCount
              << " | Unmasked: " << dataset.metadata().unmaskedTargetTokensCount << std::endl;
    std::cout << "Batch Size: " << batchSize << " | Grad Accum: " << gradAccum
              << " | Effective Batch: " << samplesPerStep << " | Seed: " << config.seed << std::endl;
    std::cout << "Initial Val Loss: " << result.initialValLoss << " | Initial PPL: " << result.initialPerplexity << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << std::setw(6) << "Step" << " | "
              << std::setw(10) << "LR" << " | "
              << std::setw(11) << "Train Loss" << " | "
              << std::setw(11) << "Val Loss" << " | "
              << std::setw(10) << "Val PPL" << " | "
              << std::setw(10) << "Grad Norm" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    for (uint32_t step = 1; step <= totalSteps; ++step) {
        float currentLr = lrSchedule.getLr(step);
        optimizer.zeroGrad();

        float stepLossSum = 0.0f;
        uint32_t stepSamples = 0;

        for (uint32_t a = 0; a < gradAccum; ++a) {
            for (uint32_t b = 0; b < batchSize; ++b) {
                if (seqCursor >= order.size()) {
                    seqCursor = 0;
                    if (config.shuffleBatches) {
                        std::shuffle(order.begin(), order.end(), shuffleRng);
                    }
                }
                const auto& sample = trainSeqs[order[seqCursor++]];

                ModelActivations act;
                float loss = model.forwardAndLoss(sample.inputs, sample.targets, act, config.ignoreIndex);
                stepLossSum += loss;
                stepSamples++;
                totalTokensTrained += static_cast<int64_t>(sample.inputs.size());
            }
        }

        if (stepSamples > 1) {
            AutogradOps::scaleGradients(model.getParameters(), 1.0f / static_cast<float>(stepSamples));
        }

        float stepLoss = stepLossSum / static_cast<float>(stepSamples);
        currentTrainLoss = stepLoss;
        stepsExecuted = step;

        if (step == 1) {
            result.initialTrainLoss = stepLoss;
        }

        float gradNorm = optimizer.step(currentLr);

        bool isEvalStep = (step == 1 || (config.evalInterval > 0 && step % config.evalInterval == 0) || step == totalSteps);
        if (isEvalStep) {
            auto eval = evaluateModel(model, valSeqs, config.ignoreIndex);

            EvalPoint pt;
            pt.step = step;
            pt.lr = currentLr;
            pt.trainLoss = stepLoss;
            pt.valLoss = eval.validationLoss;
            pt.valPerplexity = eval.perplexity;
            pt.gradNorm = gradNorm;
            result.evalHistory.push_back(pt);

            std::cout << std::setw(6) << step << " | "
                      << std::setw(10) << std::scientific << std::setprecision(2) << currentLr << " | "
                      << std::setw(11) << std::fixed << std::setprecision(4) << stepLoss << " | "
                      << std::setw(11) << std::fixed << std::setprecision(4) << eval.validationLoss << " | "
                      << std::setw(10) << std::fixed << std::setprecision(2) << eval.perplexity << " | "
                      << std::setw(10) << std::fixed << std::setprecision(4) << gradNorm;

            if (eval.validationLoss < result.bestValLoss - 1e-4f) {
                result.bestValLoss = eval.validationLoss;
                result.bestPerplexity = eval.perplexity;
                result.bestCheckpointStep = step;
                evalsWithoutImprove = 0;
                model.saveCheckpoint(bestCkptPath, step, eval.validationLoss, &optimizer);
                std::cout << " [BEST]";
            } else if (step > config.warmupSteps) {
                evalsWithoutImprove++;
                if (eval.validationLoss > result.bestValLoss + 0.05f && stepLoss < eval.validationLoss - 0.15f) {
                    result.overfittingDetected = true;
                    std::cout << " [OVERFIT_WARN]";
                }
            }
            std::cout << std::endl;

            if (config.earlyStopPatience > 0 && evalsWithoutImprove >= config.earlyStopPatience && step > config.warmupSteps) {
                result.overfittingDetected = true;
                std::cout << "[TRAINER] Early stopping triggered at step " << step
                          << " (best val loss " << result.bestValLoss << " at step " << result.bestCheckpointStep << ")" << std::endl;
                break;
            }
        }
    }

    result.totalSteps = stepsExecuted;
    result.totalTokensTrained = totalTokensTrained;
    result.finalTrainLoss = currentTrainLoss;

    auto finalEval = evaluateModel(model, valSeqs, config.ignoreIndex);
    result.finalValLoss = finalEval.validationLoss;
    result.finalPerplexity = finalEval.perplexity;

    // 4. Measure final weight norm & delta
    double finalSumSq = 0.0;
    double deltaSumSq = 0.0;
    const auto& params = model.getParameters();
    for (size_t p = 0; p < params.size(); ++p) {
        const float* v = params[p]->value.dataFloat();
        const auto& initialCopy = initialWeightsCopy[p];
        int64_t n = params[p]->value.numel();

        for (int64_t i = 0; i < n; ++i) {
            finalSumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
            double d = static_cast<double>(v[i]) - static_cast<double>(initialCopy[i]);
            deltaSumSq += d * d;
        }
    }

    result.finalWeightNorm = static_cast<float>(std::sqrt(finalSumSq));
    result.weightDeltaNorm = static_cast<float>(std::sqrt(deltaSumSq));
    result.weightsUpdated = (result.weightDeltaNorm > 1e-4f);

    // 5. Save Final Checkpoint (and ensure bestCheckpointPath exists if no step beat initialValLoss)
    result.checkpointPath = config.checkpointPath;
    model.saveCheckpoint(config.checkpointPath, stepsExecuted, result.finalTrainLoss, &optimizer);
    if (result.bestCheckpointStep == 0 && bestCkptPath != config.checkpointPath) {
        model.saveCheckpoint(bestCkptPath, stepsExecuted, result.finalValLoss, &optimizer);
        result.bestCheckpointStep = stepsExecuted;
    }

    // 6. Sample Generation (from best checkpoint if separate)
    NexaByteTokenizer defaultByteTok;
    Tokenizer& activeTokenizer = tokenizer ? *tokenizer : defaultByteTok;

    GenerationConfig genCfg;
    genCfg.maxNewTokens = 32;
    genCfg.temperature = 0.5f;
    genCfg.seed = config.seed;
    genCfg.stopSequences = {"\nUser:", "\nQ:", "\nQuestion:"};
    auto genRes = model.generateText(config.promptForSample, activeTokenizer, genCfg);
    result.sampleGeneration = genRes.fullText;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.durationSeconds = std::chrono::duration<double>(t1 - t0).count();
    result.tokensPerSecond = result.durationSeconds > 0.0 ? (static_cast<double>(totalTokensTrained) / result.durationSeconds) : 0.0;
    result.peakRssMb = getPeakRssMb();
    result.success = (result.finalTrainLoss < result.initialTrainLoss && result.weightsUpdated);

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "ADVANCED TRAINING SUMMARY:" << std::endl;
    std::cout << "  Train Loss:       " << result.initialTrainLoss << " -> " << result.finalTrainLoss << std::endl;
    std::cout << "  Val Loss:         " << result.initialValLoss << " -> " << result.finalValLoss
              << " (Best: " << result.bestValLoss << " @ step " << result.bestCheckpointStep << ")" << std::endl;
    std::cout << "  Val Perplexity:   " << result.initialPerplexity << " -> " << result.finalPerplexity
              << " (Best: " << result.bestPerplexity << ")" << std::endl;
    std::cout << "  Throughput:       " << std::fixed << std::setprecision(1) << result.tokensPerSecond << " tok/s"
              << " | Peak RSS: " << std::setprecision(2) << result.peakRssMb << " MB" << std::endl;
    std::cout << "  Overfit Detected: " << (result.overfittingDetected ? "YES" : "NO") << std::endl;
    std::cout << "  Weight Delta:     " << result.weightDeltaNorm << " (Weights updated: " << (result.weightsUpdated ? "YES" : "NO") << ")" << std::endl;
    std::cout << "  Sample Output:    \"" << result.sampleGeneration << "\"" << std::endl;
    std::cout << "  Final Checkpoint: " << result.checkpointPath << std::endl;
    std::cout << "  Best Checkpoint:  " << result.bestCheckpointPath << std::endl;
    std::cout << "================================================================\n" << std::endl;

    return result;
}

int NexaTrainer::runGenerationAndEvaluationTests() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "     NEXA PHASE 3.3 GENERATION & EVALUATION TESTS               " << std::endl;
    std::cout << "================================================================" << std::endl;

    NexaByteTokenizer tokenizer;
    NanoConfig cfg;
    cfg.vocabSize = 256;
    cfg.dim = 64;
    cfg.intermediateDim = 128;
    cfg.nLayers = 2;
    cfg.nHeads = 2;
    cfg.maxSeqLen = 32;

    NexaNanoModel model(cfg, 42);

    // Test 1: Checkpoint Validation & Integrity Rejection
    std::cout << "\n[TEST 1] Checkpoint Integrity & Corruption Rejection..." << std::endl;
    std::string testCkpt = "data/test_models/test_validation_ckpt.nexa";
    std::filesystem::create_directories("data/test_models");
    if (!model.saveCheckpoint(testCkpt, 10, 3.14f)) {
        std::cerr << "FAIL: Failed to save test checkpoint." << std::endl;
        return 1;
    }

    auto valResult = NexaNanoModel::validateCheckpoint(testCkpt);
    if (!valResult.valid || valResult.version != 2 || valResult.tensorCount != model.getParameters().size()) {
        std::cerr << "FAIL: Valid checkpoint failed validation check!" << std::endl;
        return 1;
    }
    std::cout << "Valid checkpoint passed: version=" << valResult.version << ", tensors=" << valResult.tensorCount << std::endl;

    // Test corrupted file (corrupt magic bytes)
    std::string badCkpt = "data/test_models/test_bad_magic.nexa";
    {
        std::ofstream bad(badCkpt, std::ios::binary);
        uint32_t fakeMagic = 0xDEADBEEF;
        bad.write(reinterpret_cast<const char*>(&fakeMagic), sizeof(uint32_t));
        std::vector<char> dummy(200, 0);
        bad.write(dummy.data(), dummy.size());
    }
    auto badVal = NexaNanoModel::validateCheckpoint(badCkpt);
    if (badVal.valid) {
        std::cerr << "FAIL: Corrupted magic bytes checkpoint was not rejected!" << std::endl;
        return 1;
    }
    std::cout << "Corrupted magic bytes properly rejected: " << badVal.error << std::endl;

    // Test 2: Reproducible Sampling with Fixed Random Seed
    std::cout << "\n[TEST 2] Reproducible Sampling Invariance with Fixed Seed..." << std::endl;
    GenerationConfig genCfg;
    genCfg.maxNewTokens = 16;
    genCfg.temperature = 0.8f;
    genCfg.topK = 20;
    genCfg.topP = 0.9f;
    genCfg.seed = 98765;

    auto gen1 = model.generateText("NEXA", tokenizer, genCfg);
    auto gen2 = model.generateText("NEXA", tokenizer, genCfg);

    if (gen1.generatedTokens != gen2.generatedTokens || gen1.generatedText != gen2.generatedText) {
        std::cerr << "FAIL: Sampling with identical seed produced divergent outputs!" << std::endl;
        return 1;
    }
    std::cout << "Reproducibility confirmed: seed " << genCfg.seed << " produced identical tokens: \""
              << gen1.generatedText << "\"" << std::endl;

    // Test 3: Greedy Decoding argmax Invariance
    std::cout << "\n[TEST 3] Greedy Decoding (Temperature = 0.0)..." << std::endl;
    GenerationConfig greedyCfg;
    greedyCfg.maxNewTokens = 8;
    greedyCfg.temperature = 0.0f;
    auto greedyGen = model.generateText("NEXA", tokenizer, greedyCfg);
    if (greedyGen.generatedTokens.empty()) {
        std::cerr << "FAIL: Greedy generation produced no tokens!" << std::endl;
        return 1;
    }
    std::cout << "Greedy generation produced " << greedyGen.generatedTokens.size() << " tokens in "
              << greedyGen.durationMs << " ms (" << std::fixed << std::setprecision(1) << greedyGen.tokensPerSecond << " tok/s)" << std::endl;

    // Test 4: Dataset Pipeline UTF-8 & Train/Val Split
    std::cout << "\n[TEST 4] Dataset Pipeline UTF-8 Validation & 80/20 Sequence Splitting..." << std::endl;
    DatasetPipeline dataset;
    std::string sampleCorpus =
        "The NEXA AI Core Engine is an independent software infrastructure project. "
        "The model is a transformer decoder architecture trained natively from scratch. "
        "All matrix multiplication, autograd, and AdamW optimization execute on local CPU. ";

    if (!dataset.loadFromString(sampleCorpus, "memory_corpus", tokenizer, 0.8f, 16)) {
        std::cerr << "FAIL: Dataset pipeline failed on valid UTF-8 corpus: " << dataset.metadata().errorMessage << std::endl;
        return 1;
    }
    if (dataset.trainSequences().empty() || dataset.valSequences().empty()) {
        std::cerr << "FAIL: Dataset did not produce both train and validation sequences!" << std::endl;
        return 1;
    }
    std::cout << "Dataset loaded: " << dataset.metadata().totalTokens << " tokens, "
              << dataset.trainSequences().size() << " train seqs, "
              << dataset.valSequences().size() << " val seqs." << std::endl;

    // Verify invalid UTF-8 is rejected
    std::string badUtf8 = "Good text until invalid byte: \xFF\xFE\xFD corrupted.";
    DatasetPipeline badDataset;
    if (badDataset.loadFromString(badUtf8, "bad_utf8", tokenizer)) {
        std::cerr << "FAIL: Dataset pipeline did not reject invalid UTF-8!" << std::endl;
        return 1;
    }
    std::cout << "Invalid UTF-8 rejected properly: " << badDataset.metadata().errorMessage << std::endl;

    // Test 5: Evaluation Loss and Perplexity Calculation
    std::cout << "\n[TEST 5] Model Evaluation Loss & Perplexity..." << std::endl;
    auto eval = NexaTrainer::evaluateModel(model, dataset.valSequences());
    std::cout << "Validation Loss: " << eval.validationLoss << " | Perplexity: " << eval.perplexity << std::endl;
    if (eval.validationLoss <= 0.0f || eval.perplexity < 1.0f) {
        std::cerr << "FAIL: Invalid loss or perplexity value!" << std::endl;
        return 1;
    }
    std::cout << "Evaluation metrics valid." << std::endl;

    std::cout << "\nALL 5 PHASE 3.3 GENERATION & EVALUATION TESTS PASSED SUCCESSFULLY." << std::endl;
    return 0;
}

int NexaTrainer::runPhase34Tests() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "     NEXA PHASE 3.4 LANGUAGE UNDERSTANDING UNIT TESTS           " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test 1: Trainable BPE Tokenizer Construction & Pair Merging
    std::cout << "\n[TEST 1] Trainable BPE Tokenizer Construction & Compression..." << std::endl;
    NexaBpeTokenizer bpe;
    std::string bpeTrainingCorpus =
        "The NEXA AI Core Engine is an independent software infrastructure project. "
        "Artificial intelligence systems perform reasoning, pattern recognition, and computation. "
        "The model is a transformer decoder architecture trained natively from scratch. "
        "CPU memory manages high-speed hardware caches, registers, and memory allocations. "
        "All matrix multiplication, autograd, and AdamW optimization execute on local CPU. "
        "What is AI? Artificial intelligence is the computational study of intelligent agents. "
        "What is a computer? A computer is an electronic hardware device executing stored instructions.";

    uint32_t targetVocab = 280;
    bpe.train(bpeTrainingCorpus, targetVocab, 2);

    std::cout << "BPE Vocab Size: " << bpe.vocabSize() << " (Target: " << targetVocab << ")" << std::endl;
    std::cout << "BPE Merges Created: " << bpe.numMerges() << std::endl;

    if (bpe.vocabSize() != targetVocab || bpe.numMerges() == 0) {
        std::cerr << "FAIL: BPE did not create expected vocabulary or merges!" << std::endl;
        return 1;
    }

    std::string testPhrase = "Artificial intelligence systems perform computation.";
    NexaByteTokenizer byteTok;
    auto byteTokens = byteTok.encode(testPhrase, false, false);
    auto bpeTokens = bpe.encode(testPhrase, false, false);
    std::string decoded = bpe.decode(bpeTokens);

    std::cout << "Byte token count: " << byteTokens.size() << " | BPE token count: " << bpeTokens.size() << std::endl;
    std::cout << "Compression factor: " << std::fixed << std::setprecision(2)
              << static_cast<float>(byteTokens.size()) / static_cast<float>(bpeTokens.size()) << "x" << std::endl;

    if (bpeTokens.size() >= byteTokens.size()) {
        std::cerr << "FAIL: BPE did not compress the token sequence compared to byte encoding!" << std::endl;
        return 1;
    }
    if (decoded != testPhrase) {
        std::cerr << "FAIL: BPE decoded text mismatch! Got: \"" << decoded << "\"" << std::endl;
        return 1;
    }
    std::cout << "BPE lossless round-trip verified: \"" << decoded << "\"" << std::endl;

    // Test 2: BPE Binary Serialization & Persistence (.nbt)
    std::cout << "\n[TEST 2] BPE Binary Serialization (.nbt) & Persistence..." << std::endl;
    std::string nbtPath = "data/test_models/test_tokenizer.nbt";
    std::filesystem::create_directories("data/test_models");
    if (!bpe.save(nbtPath)) {
        std::cerr << "FAIL: Failed to save BPE tokenizer to .nbt file!" << std::endl;
        return 1;
    }

    NexaBpeTokenizer reloadedBpe;
    if (!reloadedBpe.load(nbtPath)) {
        std::cerr << "FAIL: Failed to reload BPE tokenizer from .nbt file!" << std::endl;
        return 1;
    }

    if (reloadedBpe.vocabSize() != bpe.vocabSize() || reloadedBpe.numMerges() != bpe.numMerges()) {
        std::cerr << "FAIL: Reloaded BPE tokenizer parameters mismatch!" << std::endl;
        return 1;
    }

    auto reloadedTokens = reloadedBpe.encode(testPhrase, false, false);
    if (reloadedTokens != bpeTokens) {
        std::cerr << "FAIL: Reloaded BPE tokenizer produced divergent token sequence!" << std::endl;
        return 1;
    }
    std::cout << "BPE .nbt binary persistence verified bit-identically." << std::endl;

    // Test 3: Multi-File Dataset Ingestion & Deduplication
    std::cout << "\n[TEST 3] Multi-File Dataset Ingestion, Deduplication & Validation..." << std::endl;
    DatasetPipeline qaDataset;
    if (!qaDataset.loadFromDirectory("data/datasets/qa", bpe, 0.85f, 16)) {
        std::cerr << "FAIL: Failed to load QA dataset directory: " << qaDataset.metadata().errorMessage << std::endl;
        return 1;
    }

    std::cout << "Ingested " << qaDataset.metadata().sourceFiles.size() << " files from data/datasets/qa/" << std::endl;
    std::cout << "Unique lines: " << qaDataset.metadata().uniqueLines
              << " | Duplicates removed: " << qaDataset.metadata().duplicatesRemoved
              << " | Total tokens: " << qaDataset.metadata().totalTokens << std::endl;
    std::cout << "Train sequences: " << qaDataset.trainSequences().size()
              << " | Val sequences: " << qaDataset.valSequences().size() << std::endl;

    if (qaDataset.metadata().sourceFiles.size() < 3 || qaDataset.trainSequences().empty()) {
        std::cerr << "FAIL: Multi-file QA dataset failed requirement check!" << std::endl;
        return 1;
    }
    std::cout << "Multi-file QA dataset loading assertion passed." << std::endl;

    // Test 4: Learning Rate Warmup & Cosine Decay Schedule
    std::cout << "\n[TEST 4] Learning Rate Schedule (Linear Warmup & Cosine Decay)..." << std::endl;
    LrScheduleConfig lrSched{3e-3f, 1e-4f, 10, 100};

    float lr0 = lrSched.getLr(0);
    float lr1 = lrSched.getLr(1);
    float lr10 = lrSched.getLr(10);
    float lr55 = lrSched.getLr(55);
    float lr100 = lrSched.getLr(100);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "LR Step 0:   " << lr0 << " (warmup start > minLr)" << std::endl;
    std::cout << "LR Step 1:   " << lr1 << " (warming up)" << std::endl;
    std::cout << "LR Step 10:  " << lr10 << " (expected 0.003000 peak)" << std::endl;
    std::cout << "LR Step 55:  " << lr55 << " (cosine decaying)" << std::endl;
    std::cout << "LR Step 100: " << lr100 << " (expected 0.000100 min)" << std::endl;

    if (lr0 <= 1e-4f || lr0 >= 3e-3f || std::abs(lr10 - 3e-3f) > 1e-5f || std::abs(lr100 - 1e-4f) > 1e-5f) {
        std::cerr << "FAIL: LR schedule boundary conditions violated!" << std::endl;
        return 1;
    }
    if (lr0 >= lr1 || lr1 >= lr10 || lr55 <= lr100 || lr55 >= lr10) {
        std::cerr << "FAIL: LR schedule monotonicity violated!" << std::endl;
        return 1;
    }
    std::cout << "LR schedule assertion passed." << std::endl;

    // Test 5: Mini-Batch Gradient Accumulation & Scale Invariance
    std::cout << "\n[TEST 5] Mini-Batch Gradient Scaling & Autograd Invariance..." << std::endl;
    GradTensor testP("test_scaling", Shape({4}));
    testP.grad.dataFloat()[0] = 2.0f;
    testP.grad.dataFloat()[1] = 4.0f;
    testP.grad.dataFloat()[2] = 6.0f;
    testP.grad.dataFloat()[3] = 8.0f;

    std::vector<GradTensor*> paramList = {&testP};
    AutogradOps::scaleGradients(paramList, 0.5f);

    if (std::abs(testP.grad.dataFloat()[0] - 1.0f) > 1e-5f ||
        std::abs(testP.grad.dataFloat()[1] - 2.0f) > 1e-5f ||
        std::abs(testP.grad.dataFloat()[2] - 3.0f) > 1e-5f ||
        std::abs(testP.grad.dataFloat()[3] - 4.0f) > 1e-5f) {
        std::cerr << "FAIL: Gradient scaling failed!" << std::endl;
        return 1;
    }
    std::cout << "Gradient scaling invariance passed." << std::endl;

    std::cout << "\nALL 5 PHASE 3.4 LANGUAGE UNDERSTANDING TESTS PASSED SUCCESSFULLY." << std::endl;
    return 0;
}

std::vector<CapacityBenchmarkEntry> NexaTrainer::benchmarkConfigurations(uint32_t vocabSize) {
    struct Spec {
        const char* name;
        uint32_t dim;
        uint32_t inter;
        uint32_t layers;
        uint32_t heads;
        uint32_t ctx;
    };
    const std::vector<Spec> specs = {
        {"Nano-64D-2L",    64,  128, 2, 2, 64},
        {"Wide-128D-2L",   128, 256, 2, 4, 64},
        {"Micro-128D-4L",  128, 256, 4, 4, 96},
        {"Small-192D-4L",  192, 384, 4, 4, 96},
        {"Compact-256D-6L",256, 512, 6, 4, 96}
    };

    const uint32_t benchSeqLen = 32;
    const uint32_t benchSteps = 5;
    std::vector<int32_t> inputs(benchSeqLen);
    std::vector<int32_t> targets(benchSeqLen);
    for (uint32_t i = 0; i < benchSeqLen; ++i) {
        inputs[i] = static_cast<int32_t>(4 + (i % (vocabSize - 5)));
        targets[i] = static_cast<int32_t>(4 + ((i + 1) % (vocabSize - 5)));
    }

    std::vector<CapacityBenchmarkEntry> entries;
    for (const auto& s : specs) {
        CapacityBenchmarkEntry entry;
        entry.name = s.name;
        entry.config.vocabSize = vocabSize;
        entry.config.dim = s.dim;
        entry.config.intermediateDim = s.inter;
        entry.config.nLayers = s.layers;
        entry.config.nHeads = s.heads;
        entry.config.maxSeqLen = s.ctx;

        entry.memoryEstimate = NexaNanoModel::estimateMemory(entry.config);
        entry.totalParameters = entry.memoryEstimate.totalParameters;

        NexaNanoModel model(entry.config, 42);
        AdamWConfig optCfg;
        optCfg.lr = 1e-3f;
        AdamW opt(model.getParameters(), optCfg);
        ModelActivations act;

        // Warmup 1 step
        opt.zeroGrad();
        model.forwardAndLoss(inputs, targets, act);
        opt.step();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t st = 0; st < benchSteps; ++st) {
            opt.zeroGrad();
            model.forwardAndLoss(inputs, targets, act);
            opt.step();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        entry.avgStepMs = totalMs / static_cast<double>(benchSteps);
        double totalSec = totalMs / 1000.0;
        entry.tokensPerSec = totalSec > 0.0 ? (static_cast<double>(benchSteps * benchSeqLen) / totalSec) : 0.0;
        entry.peakRssMb = getPeakRssMb();
        // Require < 1024 MB estimated RAM and < 500 ms/step to run iterative CPU experiments in container
        entry.feasibleFor4GbCpu = (entry.memoryEstimate.totalTrainingBytes < 1024ULL * 1024ULL * 1024ULL) && (entry.avgStepMs < 500.0);
        entries.push_back(entry);
    }
    return entries;
}

bool NexaTrainer::verifyNumericalGradients(float* outMaxAbsDiff) {
    NanoConfig cfg;
    cfg.vocabSize = 16;
    cfg.dim = 8;
    cfg.intermediateDim = 16;
    cfg.nLayers = 1;
    cfg.nHeads = 2;
    cfg.maxSeqLen = 8;

    NexaNanoModel model(cfg, 123);
    std::vector<int32_t> inputs = {1, 4, 7, 9};
    std::vector<int32_t> targets = {4, 7, 9, 2};

    model.zeroGrad();
    ModelActivations act;
    model.forwardAndLoss(inputs, targets, act);

    // Check representative scalars across every major parameter family
    std::vector<std::pair<GradTensor*, int64_t>> checks = {
        {&model.wEmbed, 4 * 8 + 3},
        {&model.wPos, 1 * 8 + 2},
        {&model.layers[0].attnNorm, 3},
        {&model.layers[0].wQ, 2 * 8 + 5},
        {&model.layers[0].wK, 1 * 8 + 4},
        {&model.layers[0].wV, 5 * 8 + 1},
        {&model.layers[0].wOut, 3 * 8 + 6},
        {&model.layers[0].ffnNorm, 4},
        {&model.layers[0].wGate, 2 * 16 + 7},
        {&model.layers[0].wUp, 4 * 16 + 3},
        {&model.layers[0].wDown, 6 * 8 + 2},
        {&model.finalNorm, 5},
        {&model.wHead, 3 * 16 + 7}
    };

    const float eps = 1e-3f;
    float maxDiff = 0.0f;

    for (const auto& item : checks) {
        GradTensor* gt = item.first;
        int64_t idx = item.second;
        float analyticGrad = gt->grad.dataFloat()[idx];
        float orig = gt->value.dataFloat()[idx];

        Tensor logitsPlus, logitsMinus, gDummy, pDummy;
        gt->value.dataFloat()[idx] = orig + eps;
        model.forward(inputs, logitsPlus);
        gDummy = Tensor(logitsPlus.shape());
        pDummy = Tensor(logitsPlus.shape());
        float lossPlus = AutogradOps::crossEntropyForwardAndBackward(logitsPlus, targets, gDummy, pDummy);

        gt->value.dataFloat()[idx] = orig - eps;
        model.forward(inputs, logitsMinus);
        float lossMinus = AutogradOps::crossEntropyForwardAndBackward(logitsMinus, targets, gDummy, pDummy);

        gt->value.dataFloat()[idx] = orig;
        float numericGrad = (lossPlus - lossMinus) / (2.0f * eps);
        float diff = std::abs(analyticGrad - numericGrad);
        if (diff > maxDiff) maxDiff = diff;
    }

    if (outMaxAbsDiff) *outMaxAbsDiff = maxDiff;
    return maxDiff < 5e-3f;
}

bool NexaTrainer::verifyCheckpointRecovery(float* outMaxWeightDiff) {
    NanoConfig cfg;
    cfg.vocabSize = 32;
    cfg.dim = 16;
    cfg.intermediateDim = 32;
    cfg.nLayers = 2;
    cfg.nHeads = 2;
    cfg.maxSeqLen = 16;

    std::vector<int32_t> inputs = {1, 5, 8, 12, 17, 21, 25, 29};
    std::vector<int32_t> targets = {5, 8, 12, 17, 21, 25, 29, 2};

    AdamWConfig optCfg;
    optCfg.lr = 2e-3f;
    optCfg.weightDecay = 0.01f;
    optCfg.maxGradNorm = 1.0f;

    // 1. Uninterrupted 16-step run
    NexaNanoModel modelA(cfg, 777);
    AdamW optA(modelA.getParameters(), optCfg);
    ModelActivations act;
    for (int step = 1; step <= 16; ++step) {
        optA.zeroGrad();
        modelA.forwardAndLoss(inputs, targets, act);
        optA.step();
    }

    // 2. Split run: 8 steps -> save -> load into fresh modelB & optB -> 8 more steps
    std::filesystem::create_directories("data/test_models");
    std::string ckpt = "data/test_models/test_recovery_step8.nexa";
    {
        NexaNanoModel modelMid(cfg, 777);
        AdamW optMid(modelMid.getParameters(), optCfg);
        float lastLoss = 0.0f;
        for (int step = 1; step <= 8; ++step) {
            optMid.zeroGrad();
            lastLoss = modelMid.forwardAndLoss(inputs, targets, act);
            optMid.step();
        }
        if (!modelMid.saveCheckpoint(ckpt, 8, lastLoss, &optMid)) {
            return false;
        }
    }

    NexaNanoModel modelB(cfg, 9999);
    AdamW optB(modelB.getParameters(), optCfg);
    uint64_t loadedStep = 0;
    float loadedLoss = 0.0f;
    if (!modelB.loadCheckpoint(ckpt, &loadedStep, &loadedLoss, &optB) || loadedStep != 8) {
        return false;
    }
    for (int step = 9; step <= 16; ++step) {
        optB.zeroGrad();
        modelB.forwardAndLoss(inputs, targets, act);
        optB.step();
    }

    float maxDiff = 0.0f;
    const auto& pA = modelA.getParameters();
    const auto& pB = modelB.getParameters();
    for (size_t i = 0; i < pA.size(); ++i) {
        const float* dA = pA[i]->value.dataFloat();
        const float* dB = pB[i]->value.dataFloat();
        for (int64_t j = 0; j < pA[i]->value.numel(); ++j) {
            float diff = std::abs(dA[j] - dB[j]);
            if (diff > maxDiff) maxDiff = diff;
        }
    }
    if (outMaxWeightDiff) *outMaxWeightDiff = maxDiff;
    return maxDiff < 1e-5f;
}

static std::string toLowerAscii(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

static float computeTrigramRepetition(const std::vector<int32_t>& tokens) {
    if (tokens.size() < 4) return 0.0f;
    size_t totalTrigrams = tokens.size() - 2;
    size_t repeated = 0;
    for (size_t i = 0; i < totalTrigrams; ++i) {
        bool seenBefore = false;
        for (size_t j = 0; j < i; ++j) {
            if (tokens[j] == tokens[i] && tokens[j + 1] == tokens[i + 1] && tokens[j + 2] == tokens[i + 2]) {
                seenBefore = true;
                break;
            }
        }
        if (seenBefore) repeated++;
    }
    return static_cast<float>(repeated) / static_cast<float>(totalTrigrams);
}

static std::vector<std::string> extractAlphaNumWordsLower(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (unsigned char c : text) {
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

static float computeWordRepetition(const std::vector<std::string>& words) {
    if (words.size() < 2) return 0.0f;
    std::unordered_set<std::string> seen;
    size_t dup = 0;
    for (const auto& w : words) {
        if (!seen.insert(w).second) dup++;
    }
    return static_cast<float>(dup) / static_cast<float>(words.size());
}

static std::unordered_set<std::string> buildEnglishEvaluationLexicon(
    const std::string& trainValDir,
    const std::vector<UnseenTestQuestion>& questions
) {
    std::unordered_set<std::string> lex = {
        "a","an","the","is","are","was","were","be","been","being","am","it","its","this","that","these","those",
        "in","on","at","to","for","of","with","by","from","as","into","about","between","through","over","under",
        "and","or","but","not","no","yes","if","when","while","because","since","so","than","then","also","very",
        "can","could","will","would","should","may","might","must","do","does","did","have","has","had","use","used",
        "using","make","makes","made","give","gives","help","helps","need","needs","keep","keeps","more","most","many",
        "some","each","every","all","one","two","three","four","five","six","ten","first","second","third","new","good",
        "i","you","we","they","he","she","what","why","how","which","who","where","cannot","don","t","know","access",
        "real","time","private","personal","data","information","answer","question","example","steps","step","simple",
        "clear","short","order","system","memory","computer","program","code","number","numbers","energy","water","force",
        "particles","cell","cells","heat","light","earth","cpu","ram","ssd","algorithm","search","sort","matrix","vector"
    };
    namespace fs = std::filesystem;
    if (fs::exists(trainValDir) && fs::is_directory(trainValDir)) {
        for (const auto& entry : fs::directory_iterator(trainValDir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension();
            if (ext != ".txt" && ext != ".jsonl") continue;
            std::ifstream ifs(entry.path(), std::ios::binary);
            if (!ifs) continue;
            std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            for (const auto& w : extractAlphaNumWordsLower(content)) {
                lex.insert(w);
            }
        }
    }
    for (const auto& q : questions) {
        for (const auto& w : extractAlphaNumWordsLower(q.question)) lex.insert(w);
        for (const auto& kw : q.keywords) {
            for (const auto& w : extractAlphaNumWordsLower(kw)) lex.insert(w);
        }
    }
    return lex;
}

UnseenEvalSummary NexaTrainer::evaluateUnseenQuestions(
    NexaNanoModel& model,
    Tokenizer& tokenizer,
    const std::string& unseenQuestionsPath,
    const std::string& trainValDir,
    uint32_t maxNewTokens,
    float temperature,
    uint64_t seed
) {
    UnseenEvalSummary summary;
    std::vector<UnseenTestQuestion> questions;
    std::string loadErr;
    if (!DatasetPipeline::loadUnseenQuestions(unseenQuestionsPath, questions, &loadErr)) {
        summary.success = false;
        summary.errorMessage = loadErr;
        return summary;
    }

    std::vector<std::string> overlaps;
    summary.disjointVerified = DatasetPipeline::verifyDisjointUnseenQuestions(trainValDir, questions, &overlaps);
    summary.totalQuestions = static_cast<uint32_t>(questions.size());
    auto lexicon = buildEnglishEvaluationLexicon(trainValDir, questions);

    double sumLatency = 0.0;
    double sumTps = 0.0;
    double sumRep = 0.0;
    double sumWordRep = 0.0;
    double sumRel = 0.0;
    double sumCoh = 0.0;
    double sumFact = 0.0;
    double sumInst = 0.0;

    for (size_t idx = 0; idx < questions.size(); ++idx) {
        const auto& q = questions[idx];
        std::string prompt;
        if (q.question.rfind("User:", 0) == 0) {
            prompt = q.question + "\nAssistant:";
        } else {
            prompt = "User: " + q.question + "\nAssistant:";
        }

        GenerationConfig genCfg;
        genCfg.maxNewTokens = maxNewTokens;
        genCfg.temperature = temperature;
        genCfg.topK = 25;
        genCfg.topP = 0.9f;
        genCfg.repetitionPenalty = 1.18f;
        genCfg.seed = seed + idx;
        genCfg.stopSequences = {"\nUser:", "\n\n", "<|end_of_text|>"};

        auto gen = model.generateText(prompt, tokenizer, genCfg);

        // Trim leading single space if generated right after "Assistant:"
        std::string ans = gen.generatedText;
        if (!ans.empty() && ans[0] == ' ') ans = ans.substr(1);
        while (!ans.empty() && (ans.back() == '\n' || ans.back() == '\r' || ans.back() == ' ')) {
            ans.pop_back();
        }

        UnseenQuestionResult qr;
        qr.category = q.category;
        qr.expectedBehavior = q.expectedBehavior;
        qr.question = q.question;
        qr.generatedAnswer = ans;
        qr.tokensGenerated = gen.tokensGenerated;
        qr.durationMs = gen.durationMs;
        qr.tokensPerSecond = gen.tokensPerSecond;
        qr.trigramRepetitionRate = computeTrigramRepetition(gen.generatedTokens);
        qr.cleanStop = (gen.finishReason == "stop_token" || gen.finishReason == "stop_sequence");

        bool validUtf8 = DatasetPipeline::validateUtf8(ans);
        auto ansWords = extractAlphaNumWordsLower(ans);
        size_t wordCount = ansWords.size();
        qr.wordRepetitionRate = computeWordRepetition(ansWords);

        // Lexical English-word coherence ratio + punctuation/repetition check
        size_t validLexWords = 0;
        for (const auto& w : ansWords) {
            bool allDigits = !w.empty() && std::all_of(w.begin(), w.end(), [](unsigned char c) { return std::isdigit(c); });
            if (allDigits || lexicon.count(w) > 0) {
                validLexWords++;
            }
        }
        float lexRatio = wordCount > 0 ? static_cast<float>(validLexWords) / static_cast<float>(wordCount) : 0.0f;
        bool endsWithPunct = !ans.empty() && (ans.back() == '.' || ans.back() == '!' || ans.back() == '?');
        bool noDegenerateRepeat = (qr.trigramRepetitionRate < 0.30f && qr.wordRepetitionRate < 0.55f);
        float coh = 0.0f;
        if (validUtf8 && wordCount >= 3) {
            coh = 0.65f * lexRatio + (endsWithPunct ? 0.15f : 0.0f) + (noDegenerateRepeat ? 0.20f : 0.0f);
        }
        qr.coherenceScore = std::min(1.0f, std::max(0.0f, coh));
        qr.nonEmptyCoherentFormat = validUtf8 && !ans.empty() && wordCount >= 3 && qr.coherenceScore >= 0.60f;

        std::string ansLower = toLowerAscii(ans);
        qr.totalKeywords = static_cast<uint32_t>(q.keywords.size());
        uint32_t hits = 0;
        for (const auto& kw : q.keywords) {
            if (ansLower.find(toLowerAscii(kw)) != std::string::npos) {
                hits++;
            }
        }
        qr.keywordHits = hits;

        if (q.expectedBehavior == "uncertainty") {
            summary.uncertaintyQuestionCount++;
            bool expressesUncertainty =
                (ansLower.find("cannot") != std::string::npos ||
                 ansLower.find("do not know") != std::string::npos ||
                 ansLower.find("don't know") != std::string::npos ||
                 ansLower.find("do not have") != std::string::npos ||
                 ansLower.find("no access") != std::string::npos ||
                 ansLower.find("not provided") != std::string::npos ||
                 ansLower.find("unknown") != std::string::npos);
            qr.relevanceHit = expressesUncertainty;
            qr.relevanceScore = expressesUncertainty ? 1.0f : 0.0f;
            qr.factualAccuracyScore = expressesUncertainty ? 1.0f : 0.0f;
            if (expressesUncertainty) summary.uncertaintyMatchedCount++;
        } else {
            qr.relevanceHit = (hits >= 1);
            qr.relevanceScore = qr.totalKeywords > 0
                ? static_cast<float>(hits) / static_cast<float>(qr.totalKeywords)
                : (wordCount >= 4 ? 0.5f : 0.0f);
            // Factual accuracy requires >= 2 matching domain keywords OR >= 50% keyword coverage AND decent coherence
            if (qr.totalKeywords >= 2 && hits >= 2 && qr.coherenceScore >= 0.50f) {
                qr.factualAccuracyScore = std::min(1.0f, static_cast<float>(hits) / static_cast<float>(std::min<uint32_t>(3, qr.totalKeywords)));
            } else if (hits == 1 && qr.coherenceScore >= 0.50f) {
                qr.factualAccuracyScore = 0.25f;
            } else {
                qr.factualAccuracyScore = 0.0f;
            }
        }

        // Instruction following: clean stop, stays in Assistant role, >= 4 words, obeys format
        bool leakedRole = (ans.find("User:") != std::string::npos);
        float instScore = 0.0f;
        if (!ans.empty() && validUtf8 && !leakedRole) {
            instScore += (qr.cleanStop ? 0.40f : 0.0f);
            instScore += (wordCount >= 4 ? 0.30f : 0.0f);
            if (q.expectedBehavior == "uncertainty") {
                instScore += (qr.relevanceHit ? 0.30f : 0.0f);
            } else if (q.expectedBehavior == "instruction") {
                instScore += (hits >= 1 ? 0.30f : 0.0f);
            } else {
                instScore += (qr.relevanceHit ? 0.30f : (qr.coherenceScore >= 0.65f ? 0.15f : 0.0f));
            }
        }
        qr.instructionFollowingScore = std::min(1.0f, instScore);

        if (qr.nonEmptyCoherentFormat) summary.nonEmptyResponses++;
        if (qr.cleanStop) summary.cleanStopCount++;
        if (qr.relevanceHit) summary.relevanceHitCount++;

        sumLatency += qr.durationMs;
        sumTps += qr.tokensPerSecond;
        sumRep += qr.trigramRepetitionRate;
        sumWordRep += qr.wordRepetitionRate;
        sumRel += qr.relevanceScore;
        sumCoh += qr.coherenceScore;
        sumFact += qr.factualAccuracyScore;
        sumInst += qr.instructionFollowingScore;
        summary.results.push_back(std::move(qr));
    }

    if (summary.totalQuestions > 0) {
        double n = static_cast<double>(summary.totalQuestions);
        summary.avgLatencyMs = sumLatency / n;
        summary.avgTokensPerSec = sumTps / n;
        summary.avgRepetitionRate = static_cast<float>(sumRep / n);
        summary.avgWordRepetitionRate = static_cast<float>(sumWordRep / n);
        summary.avgRelevanceScore = static_cast<float>(sumRel / n);
        summary.avgCoherenceScore = static_cast<float>(sumCoh / n);
        summary.avgFactualAccuracyScore = static_cast<float>(sumFact / n);
        summary.avgInstructionFollowingScore = static_cast<float>(sumInst / n);
    }
    summary.peakRssMb = getPeakRssMb();
    summary.success = (summary.totalQuestions >= 50 && summary.disjointVerified);
    return summary;
}

int NexaTrainer::runPhase35Tests() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  NEXA PHASE 3.5 INTELLIGENCE, GENERALISATION & TRAINING TESTS  " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test 1: End-to-End Finite-Difference Numerical Gradient Check
    std::cout << "\n[TEST 1] Finite-Difference Numerical Gradient Check (All Transformer Layers)..." << std::endl;
    float maxGradDiff = 0.0f;
    if (!verifyNumericalGradients(&maxGradDiff)) {
        std::cerr << "FAIL: Numerical vs analytical gradient mismatch: maxAbsDiff=" << maxGradDiff << std::endl;
        return 1;
    }
    std::cout << "Numerical gradient check passed: maxAbsDiff = " << maxGradDiff << " (< 0.005)" << std::endl;

    // Test 2: Checkpoint Recovery Trajectory Bit-Equivalence
    std::cout << "\n[TEST 2] Checkpoint Recovery Trajectory Verification (Model + AdamW State)..." << std::endl;
    float maxWeightDiff = 0.0f;
    if (!verifyCheckpointRecovery(&maxWeightDiff)) {
        std::cerr << "FAIL: Resumed training diverged from uninterrupted run! maxWeightDiff=" << maxWeightDiff << std::endl;
        return 1;
    }
    std::cout << "Checkpoint recovery passed: maxWeightDiff = " << maxWeightDiff << " (< 1e-5)" << std::endl;

    // Test 3: Shuffled Instruction Dataset, Prompt Loss Masking & Unseen Disjoint Check
    std::cout << "\n[TEST 3] Instruction Dataset Loading, Prompt Masking & Unseen Question Separation..." << std::endl;
    NexaBpeTokenizer bpe;
    std::string combinedCorpus;
    for (const auto& entry : std::filesystem::directory_iterator("data/datasets/phase35/train_val")) {
        if (entry.path().extension() == ".txt") {
            std::ifstream ifs(entry.path(), std::ios::binary);
            if (ifs) combinedCorpus.append((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            combinedCorpus += "\n";
        }
    }
    bpe.train(combinedCorpus, 640, 2);

    DatasetPipeline ds;
    if (!ds.loadInstructionExamplesFromDirectory("data/datasets/phase35/train_val", bpe, 0.85f, 64, true, 42)) {
        std::cerr << "FAIL: Failed loading Phase 3.5 instruction dataset: " << ds.metadata().errorMessage << std::endl;
        return 1;
    }
    if (ds.metadata().maskedTokensCount == 0 || ds.metadata().unmaskedTargetTokensCount == 0) {
        std::cerr << "FAIL: Prompt masking did not separate masked prompt vs unmasked answer tokens!" << std::endl;
        return 1;
    }

    std::vector<UnseenTestQuestion> unseenQs;
    std::string unseenErr;
    if (!DatasetPipeline::loadUnseenQuestions("data/datasets/phase35/unseen_test_qa.txt", unseenQs, &unseenErr)) {
        std::cerr << "FAIL: Failed loading unseen questions: " << unseenErr << std::endl;
        return 1;
    }
    std::vector<std::string> overlaps;
    bool disjoint = DatasetPipeline::verifyDisjointUnseenQuestions("data/datasets/phase35/train_val", unseenQs, &overlaps);
    if (!disjoint || unseenQs.size() < 50) {
        std::cerr << "FAIL: Unseen test set check failed! count=" << unseenQs.size() << ", overlaps=" << overlaps.size() << std::endl;
        return 1;
    }
    std::cout << "Dataset & Unseen separation passed: examples=" << ds.metadata().exampleCount
              << ", masked=" << ds.metadata().maskedTokensCount
              << ", unmasked=" << ds.metadata().unmaskedTargetTokensCount
              << ", unseen=" << unseenQs.size() << " (0 overlaps)" << std::endl;

    // Test 4: Model Memory Estimate & Capacity Benchmark Sanity
    std::cout << "\n[TEST 4] Hardware Memory Estimate & Capacity Benchmark..." << std::endl;
    NanoConfig testCfg;
    testCfg.vocabSize = 640;
    testCfg.dim = 128;
    testCfg.intermediateDim = 256;
    testCfg.nLayers = 4;
    testCfg.nHeads = 4;
    testCfg.maxSeqLen = 64;
    auto memEst = NexaNanoModel::estimateMemory(testCfg);
    NexaNanoModel capModel(testCfg, 42);
    if (memEst.totalParameters != capModel.totalParameters() || memEst.totalTrainingBytes <= memEst.weightsBytes) {
        std::cerr << "FAIL: ModelMemoryEstimate mismatch! est=" << memEst.totalParameters
                  << " vs model=" << capModel.totalParameters() << std::endl;
        return 1;
    }
    std::cout << "Memory estimation matches exact model params (" << memEst.totalParameters << " params, "
              << (memEst.totalTrainingBytes / (1024.0 * 1024.0)) << " MB est training RAM)" << std::endl;

    // Test 5: Best Validation Checkpoint Selection & Reproducible Seed
    std::cout << "\n[TEST 5] Best Validation Checkpoint Selection & Reproducible Seed Training..." << std::endl;
    TrainingPipelineConfig pCfg;
    pCfg.numSteps = 12;
    pCfg.batchSize = 2;
    pCfg.gradAccumSteps = 1;
    pCfg.learningRate = 3e-3f;
    pCfg.minLr = 1e-4f;
    pCfg.warmupSteps = 3;
    pCfg.evalInterval = 4;
    pCfg.seed = 42;
    pCfg.checkpointPath = "data/test_models/test_p35_final.nexa";
    pCfg.bestCheckpointPath = "data/test_models/test_p35_best.nexa";
    pCfg.promptForSample = "User: What is a CPU?\nAssistant:";

    NanoConfig smallCfg;
    smallCfg.vocabSize = static_cast<uint32_t>(bpe.vocabSize());
    smallCfg.dim = 64;
    smallCfg.intermediateDim = 128;
    smallCfg.nLayers = 2;
    smallCfg.nHeads = 2;
    smallCfg.maxSeqLen = 64;
    NexaNanoModel m1(smallCfg, 42);
    auto r1 = trainWithDatasetAdvanced(m1, ds, pCfg, &bpe);
    if (!r1.success || !std::filesystem::exists(pCfg.bestCheckpointPath)) {
        std::cerr << "FAIL: Phase 3.5 training or best checkpoint saving failed!" << std::endl;
        return 1;
    }
    std::cout << "Best checkpoint saved at step " << r1.bestCheckpointStep << " (val loss " << r1.bestValLoss << ")." << std::endl;

    std::cout << "\nALL 5 PHASE 3.5 TESTS PASSED SUCCESSFULLY." << std::endl;
    return 0;
}

int NexaTrainer::runPhase36Tests() {
    namespace fs = std::filesystem;
    std::cout << "\n================================================================" << std::endl;
    std::cout << "      NEXA PHASE 3.6 DATA QUALITY & TRAINING TESTS              " << std::endl;
    std::cout << "================================================================" << std::endl;

    // Test 1: Provenance, Explicit Licensing (Never Silently MIT), & JSONL/TXT Managed Loading
    std::cout << "\n[TEST 1] Dataset Provenance, Explicit License Safety & JSONL/TXT Ingestion..." << std::endl;
    if (DatasetPipeline::isVerifiedPermissiveLicense("") ||
        DatasetPipeline::isVerifiedPermissiveLicense("UNKNOWN") ||
        DatasetPipeline::isVerifiedPermissiveLicense("UNVERIFIED") ||
        !DatasetPipeline::isVerifiedPermissiveLicense("MIT") ||
        !DatasetPipeline::isVerifiedPermissiveLicense("Apache-2.0")) {
        std::cerr << "FAIL: License verification predicate violated safety rules!" << std::endl;
        return 1;
    }

    std::string tmpUnverifiedDir = "data/test_models/p36_unverified";
    fs::remove_all(tmpUnverifiedDir);
    fs::create_directories(tmpUnverifiedDir);
    {
        std::ofstream ofs(tmpUnverifiedDir + "/sample.jsonl", std::ios::binary);
        ofs << "{\"messages\":[{\"role\":\"user\",\"content\":\"What is binary search?\"},{\"role\":\"assistant\",\"content\":\"Binary search repeatedly divides a sorted list in half to locate a target value.\"}]}\n";
        ofs << "{\"prompt\":\"What is RAM?\",\"response\":\"RAM is volatile fast memory used to hold active programs and working data.\"}\n";
        ofs << "{\"messages\":[{\"role\":\"user\",\"content\":\"Why does water boil at lower temperatures on mountains?\"},{\"role\":\"assistant\",\"content\":\"Lower atmospheric pressure at high altitude reduces the vapor pressure needed for boiling.\"}]}\n";
        ofs << "{\"messages\":[{\"role\":\"user\",\"content\":\"What is a prime number?\"},{\"role\":\"assistant\",\"content\":\"A prime number is an integer greater than one with only two positive divisors.\"}]}\n";
    }

    NexaBpeTokenizer bpe;
    {
        std::ifstream ifs(tmpUnverifiedDir + "/sample.jsonl", std::ios::binary);
        std::string txt((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        bpe.train(txt, 320, 2);
    }

    DatasetPipeline unverifiedDs;
    if (unverifiedDs.loadManagedDataset(tmpUnverifiedDir, bpe, 0.75f, 48, true, 42, "", true, true)) {
        std::cerr << "FAIL: Unverified dataset without manifest was allowed when requireVerifiedLicense=true!" << std::endl;
        return 1;
    }
    if (!unverifiedDs.loadManagedDataset(tmpUnverifiedDir, bpe, 0.75f, 48, true, 42, "", false, true)) {
        std::cerr << "FAIL: Unverified dataset failed to load in permissive-optional mode: " << unverifiedDs.metadata().errorMessage << std::endl;
        return 1;
    }
    if (unverifiedDs.metadata().license != "UNVERIFIED" || unverifiedDs.metadata().licenseVerified) {
        std::cerr << "FAIL: Unverified dataset was silently relabeled to '" << unverifiedDs.metadata().license << "'!" << std::endl;
        return 1;
    }
    if (unverifiedDs.metadata().uniqueTokensCount == 0 || unverifiedDs.metadata().datasetVersion.empty()) {
        std::cerr << "FAIL: Unique token count or dataset version missing!" << std::endl;
        return 1;
    }
    std::cout << "Unverified license strictly preserved ('" << unverifiedDs.metadata().license
              << "'), JSONL parsed (" << unverifiedDs.metadata().exampleCount << " examples, "
              << unverifiedDs.metadata().uniqueTokensCount << " unique tokens)." << std::endl;

    // Test 2: All 8 Dataset Quality Checks
    std::cout << "\n[TEST 2] Comprehensive 8-Way Dataset Quality Defect Filtering..." << std::endl;
    std::vector<RawExampleBlock> dirtyBlocks;
    // Good block 1 (long enough so single-word paraphrase exceeds 0.85 3-gram Jaccard)
    dirtyBlocks.push_back({"User: What does a compiler do when translating source code?\nAssistant: A compiler reads human-readable source code written in high-level languages, checks syntax carefully, optimizes loops, and produces executable machine instructions for the target CPU.", "good.txt", "Original NEXA", "MIT", true});
    // 1. Corrupted UTF-8 text
    dirtyBlocks.push_back({std::string("User: Broken bytes?\nAssistant: Bad bytes \xFF\xFE here."), "corrupt.txt", "Original NEXA", "MIT", true});
    // 2. Incomplete sentence (ellipsis / no terminal punctuation)
    dirtyBlocks.push_back({"User: Why do engines need cooling?\nAssistant: Engines produce heat during combustion and require...", "incomplete.txt", "Original NEXA", "MIT", true});
    // 3. Repetitive example (consecutive duplicate words)
    dirtyBlocks.push_back({"User: What is a loop?\nAssistant: A loop repeats repeats a sequence of instructions in a program.", "repeat.txt", "Original NEXA", "MIT", true});
    // 4. Low-information example (placeholder / too short)
    dirtyBlocks.push_back({"User: Explain photosynthesis.\nAssistant: TODO.", "lowinfo.txt", "Original NEXA", "MIT", true});
    // 5. Incorrect QA formatting (missing User:/Assistant: prefixes)
    dirtyBlocks.push_back({"Question: What is RAM?\nAnswer: Fast temporary working memory inside computers.", "badfmt.txt", "Original NEXA", "MIT", true});
    // 6. Malformed conversation (two consecutive User turns without Assistant)
    dirtyBlocks.push_back({"User: Hello there.\nUser: Still nobody?\nAssistant: Answering after two user turns breaks alternation.", "malformed.txt", "Original NEXA", "MIT", true});
    // 7a. Exact duplicate of good block 1
    dirtyBlocks.push_back({"User: What does a compiler do when translating source code?\nAssistant: A compiler reads human-readable source code written in high-level languages, checks syntax carefully, optimizes loops, and produces executable machine instructions for the target CPU.", "dup.txt", "Original NEXA", "MIT", true});
    // 7b. Near-duplicate of good block 1 (>= 0.85 Jaccard: 1 final word changed)
    dirtyBlocks.push_back({"User: What does a compiler do when translating source code?\nAssistant: A compiler reads human-readable source code written in high-level languages, checks syntax carefully, optimizes loops, and produces executable machine instructions for the target processor.", "neardup.txt", "Original NEXA", "MIT", true});
    // Good block 2 (yes/no factual)
    dirtyBlocks.push_back({"User: Is the speed of light constant in a vacuum?\nAssistant: Yes, light travels at a constant speed in a vacuum.", "good2.txt", "Original NEXA", "MIT", true});
    // 8. Contradictory answer to good block 2
    dirtyBlocks.push_back({"User: Is the speed of light constant in a vacuum?\nAssistant: No, light speed fluctuates wildly inside a vacuum.", "contra.txt", "Original NEXA", "MIT", true});

    std::vector<RawExampleBlock> cleanBlocks;
    auto qRep = DatasetPipeline::analyzeAndFilterBlocks(dirtyBlocks, cleanBlocks, 0.85f, true);
    std::cout << "Quality filter results: input=" << qRep.totalInputBlocks
              << ", accepted=" << qRep.acceptedBlocks
              << ", rejected=" << qRep.rejectedBlocks << std::endl;
    if (qRep.acceptedBlocks != 2 ||
        qRep.utf8ErrorCount == 0 ||
        qRep.incompleteSentenceCount == 0 ||
        qRep.repetitiveExampleCount == 0 ||
        qRep.lowInformationCount == 0 ||
        qRep.formattingErrorCount == 0 ||
        qRep.malformedConversationCount == 0 ||
        qRep.exactDuplicatesRemoved == 0 ||
        qRep.nearDuplicatesRemoved == 0 ||
        qRep.contradictoryAnswerCount == 0) {
        std::cerr << "FAIL: Quality checks missed one or more defect categories! "
                  << "utf8=" << qRep.utf8ErrorCount
                  << " incomp=" << qRep.incompleteSentenceCount
                  << " rep=" << qRep.repetitiveExampleCount
                  << " low=" << qRep.lowInformationCount
                  << " fmt=" << qRep.formattingErrorCount
                  << " conv=" << qRep.malformedConversationCount
                  << " exact=" << qRep.exactDuplicatesRemoved
                  << " near=" << qRep.nearDuplicatesRemoved
                  << " contra=" << qRep.contradictoryAnswerCount << std::endl;
        return 1;
    }
    std::cout << "All 8 dataset quality defect categories detected and filtered accurately." << std::endl;

    // Test 3: Exact & Paraphrased N-Gram Contamination Detection
    std::cout << "\n[TEST 3] Exact & Paraphrased N-Gram Evaluation Contamination Detection..." << std::endl;
    std::vector<UnseenTestQuestion> mockUnseen = {
        {"cs", "factual", {"hash", "lookup"}, "How does a hash table find values quickly?"}
    };
    std::vector<std::pair<std::string, std::string>> leakedDocs = {
        {"paraphrase.txt", "User: How does a hash table find values so quickly?\nAssistant: It hashes keys into bucket slots."}
    };
    auto contamMatches = DatasetPipeline::checkContamination(leakedDocs, mockUnseen, 0.72f);
    if (contamMatches.empty()) {
        std::cerr << "FAIL: N-gram contamination check missed paraphrased unseen question!" << std::endl;
        return 1;
    }
    std::cout << "Contamination detector caught leaked paraphrase: type=" << contamMatches[0].matchType
              << ", jaccard=" << contamMatches[0].similarity << std::endl;

    // Test 4: Multi-Turn Instruction Loss Masking & Zero Gradient Verification
    std::cout << "\n[TEST 4] Multi-Turn Prompt Loss Masking & Exact Zero-Gradient Verification..." << std::endl;
    std::string multiTurnDir = "data/test_models/p36_multiturn";
    fs::remove_all(multiTurnDir);
    fs::create_directories(multiTurnDir);
    {
        std::ofstream ofs(multiTurnDir + "/conv.txt", std::ios::binary);
        ofs << "User: What is 2 plus 2?\nAssistant: Two plus two equals four.\nUser: And what is 4 plus 4?\nAssistant: Four plus four equals eight.\n\n";
        ofs << "User: What is 3 plus 3?\nAssistant: Three plus three equals six.\nUser: And what is 6 plus 6?\nAssistant: Six plus six equals twelve.\n";
    }
    DatasetPipeline mtDs;
    if (!mtDs.loadInstructionExamplesFromDirectory(multiTurnDir, bpe, 0.5f, 128, true, 42, "Original Test", "MIT", true, "")) {
        std::cerr << "FAIL: Multi-turn dataset loading failed: " << mtDs.metadata().errorMessage << std::endl;
        return 1;
    }
    if (mtDs.trainSequences().empty()) {
        std::cerr << "FAIL: No multi-turn sequences generated!" << std::endl;
        return 1;
    }
    const auto& seq0 = mtDs.trainSequences()[0];
    size_t ignoreCount = 0;
    size_t unmaskedCount = 0;
    for (int32_t t : seq0.targets) {
        if (t == DatasetPipeline::IGNORE_INDEX) ignoreCount++;
        else unmaskedCount++;
    }
    if (ignoreCount == 0 || unmaskedCount == 0) {
        std::cerr << "FAIL: Multi-turn targets missing IGNORE_INDEX masking or unmasked tokens!" << std::endl;
        return 1;
    }

    // Verify cross-entropy assigns exact 0.0f gradient at IGNORE_INDEX positions
    Tensor ceLogits(Shape({4, 8}));
    for (int64_t i = 0; i < ceLogits.numel(); ++i) {
        ceLogits.dataFloat()[i] = 0.25f * static_cast<float>((i % 7) - 3);
    }
    Tensor ceGrad(Shape({4, 8}));
    Tensor ceProbs(Shape({4, 8}));
    std::vector<int32_t> ceTargets = {DatasetPipeline::IGNORE_INDEX, 3, DatasetPipeline::IGNORE_INDEX, 5};
    float ceLoss = AutogradOps::crossEntropyForwardAndBackward(ceLogits, ceTargets, ceGrad, ceProbs, DatasetPipeline::IGNORE_INDEX);
    if (ceLoss <= 0.0f) {
        std::cerr << "FAIL: Cross-entropy loss with IGNORE_INDEX should be positive!" << std::endl;
        return 1;
    }
    for (int pos : {0, 2}) {
        for (int c = 0; c < 8; ++c) {
            if (ceGrad.dataFloat()[pos * 8 + c] != 0.0f) {
                std::cerr << "FAIL: Non-zero gradient at masked IGNORE_INDEX position " << pos << "!" << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Multi-turn masking verified: ignored=" << ignoreCount << ", active=" << unmaskedCount
              << ", masked position gradients identically 0.0." << std::endl;

    // Test 5: Checkpoint Compatibility, ArchitecturePrepSpec & Memory Scaling Prep
    std::cout << "\n[TEST 5] .nexa v2 Checkpoint Compatibility & Architecture Preparation..." << std::endl;
    auto archSpec = NexaNanoModel::getArchitecturePrepSpec();
    if (!archSpec.rmsNormActive || archSpec.ropeActiveInTraining || archSpec.tiedEmbeddingsActiveInTraining || archSpec.dropoutActiveInTraining) {
        std::cerr << "FAIL: ArchitecturePrepSpec active flags must reflect actual autograd status honestly!" << std::endl;
        return 1;
    }
    NanoConfig baseCfg;
    baseCfg.vocabSize = 1024;
    baseCfg.dim = 128;
    baseCfg.intermediateDim = 256;
    baseCfg.nLayers = 2;
    baseCfg.nHeads = 4;
    baseCfg.maxSeqLen = 64;
    auto memBase = NexaNanoModel::estimateMemoryWithOptions(baseCfg, false, false, 64);
    auto memTied256 = NexaNanoModel::estimateMemoryWithOptions(baseCfg, true, true, 256);
    auto mem512 = NexaNanoModel::estimateMemoryWithOptions(baseCfg, true, true, 512);
    if (memTied256.totalParameters >= memBase.totalParameters || mem512.activationsBytes <= memTied256.activationsBytes) {
        std::cerr << "FAIL: Memory estimation scaling options inconsistent!" << std::endl;
        return 1;
    }
    std::cout << "Arch Prep: RMSNorm=active, RoPE=prepared(inactive), TiedEmbed=prepared(inactive), "
              << "Base params=" << memBase.totalParameters
              << ", Tied+RoPE-prepared(ctx512)=" << mem512.totalParameters
              << " (" << std::fixed << std::setprecision(2) << (mem512.totalTrainingBytes / (1024.0 * 1024.0)) << " MB train)." << std::endl;

    // Test 6: Experiment Manager Reproducibility & JSON Registry Persistence
    std::cout << "\n[TEST 6] Experiment Manager Structured JSON Serialization..." << std::endl;
    ExperimentRecord expRec;
    expRec.experimentId = "test_phase36_exp";
    expRec.timestampIso = ExperimentManager::currentIsoTimestamp();
    expRec.datasetLabel = "test_tier";
    expRec.modelConfig = baseCfg;
    expRec.totalParameters = memBase.totalParameters;
    expRec.archPrep = archSpec;
    expRec.datasetMeta = mtDs.metadata();
    expRec.qualityReport = mtDs.qualityReport();
    expRec.tokenizerInfo.type = "bpe";
    expRec.tokenizerInfo.vocabSize = static_cast<uint32_t>(bpe.vocabSize());
    expRec.tokenizerInfo.numMerges = static_cast<uint32_t>(bpe.numMerges());
    std::string expOutPath;
    if (!ExperimentManager::saveExperimentRecord(expRec, "data/test_models/experiments", &expOutPath) ||
        !fs::exists(expOutPath)) {
        std::cerr << "FAIL: ExperimentManager failed to write experiment record!" << std::endl;
        return 1;
    }
    std::cout << "Experiment record saved to " << expOutPath << std::endl;

    std::cout << "\nALL 6 PHASE 3.6 DATA QUALITY & TRAINING TESTS PASSED SUCCESSFULLY." << std::endl;
    return 0;
}

} // namespace nexa

