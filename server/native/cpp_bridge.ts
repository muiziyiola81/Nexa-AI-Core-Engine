/**
 * NEXA Native Runtime - Process-Based C++ Runtime Bridge
 * Executes the compiled C++20 nexa-runtime-cli binary via safe process execution.
 * Provides strict timeout handling, process lifecycle management, stderr capture,
 * and safe argument passing with NO shell interpolation and NO silent fallback.
 */

import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

export interface CppRuntimeInfo {
  engine: string;
  version: string;
  standard: string;
  arch: string;
  status: string;
  binaryPath: string;
}

export interface CppTensorTestResult {
  status: string;
  runtime: string;
  tests_passed: number;
  matmul: {
    result: number[][];
    passed: boolean;
  };
  softmax: {
    sum: number;
    passed: boolean;
  };
  rmsnorm: {
    val: number;
    passed: boolean;
  };
  tokenizer: {
    token_count: number;
    passed: boolean;
  };
}

export interface CppForwardPassResult {
  status: string;
  runtime: string;
  dim: number;
  seq_len: number;
  duration_us: number;
  output_checksum: number;
  tensor_elements: number;
}

export interface CppTokenizeResult {
  status: string;
  tokenizer?: string;
  vocab_size?: number;
  tokens: number[];
  count: number;
  error?: string;
}

export interface CppInspectResult {
  status: string;
  valid: boolean;
  version?: number;
  architecture: string;
  vocab_size?: number;
  hidden_dim: number;
  intermediate_dim?: number;
  layers: number;
  heads: number;
  kv_heads: number;
  max_seq_len?: number;
  rope_theta?: number;
  use_rope?: boolean;
  tie_embeddings?: boolean;
  tensors: number;
  weight_bytes: number;
}

export interface CppTrainingTestResult {
  status: string;
  tests_passed: number;
}

export interface CppTrainingRunResult {
  status: string;
  engine: string;
  total_steps: number;
  initial_loss: number;
  final_loss: number;
  loss_reduction_pct: number;
  initial_weight_norm: number;
  final_weight_norm: number;
  weight_delta_norm: number;
  weights_updated: boolean;
  checkpoint_verified: boolean;
  reloaded_output_identical: boolean;
  checkpoint_path: string;
  duration_s: number;
}

export interface CppGenerationResult {
  status: string;
  checkpoint: string;
  prompt: string;
  generated_text: string;
  full_text: string;
  prompt_tokens?: number;
  tokens_generated: number;
  max_seq_len?: number;
  vocab_size?: number;
  use_rope?: boolean;
  tie_embeddings?: boolean;
  duration_ms: number;
  tokens_per_second: number;
  finish_reason: string;
  error?: string;
}

export interface CppCheckpointValidationResult {
  status: string;
  valid: boolean;
  version: number;
  arch_type: number;
  vocab_size: number;
  dim: number;
  intermediate_dim?: number;
  layers: number;
  heads: number;
  max_seq_len?: number;
  rope_theta?: number;
  use_rope?: boolean;
  tie_embeddings?: boolean;
  tokenizer_hash?: number;
  tensors: number;
  weight_bytes: number;
  step: number;
  loss: number;
  has_optimizer_state: boolean;
  error?: string;
}

export interface CppDatasetTrainingResult {
  status: string;
  total_steps: number;
  initial_train_loss: number;
  final_train_loss: number;
  initial_val_loss: number;
  final_val_loss: number;
  initial_perplexity: number;
  final_perplexity: number;
  initial_weight_norm: number;
  final_weight_norm: number;
  weight_delta_norm: number;
  weights_updated: boolean;
  checkpoint_path: string;
  sample_generation: string;
  duration_s: number;
  error?: string;
}

export interface CppGenerationTestResult {
  status: string;
  tests_passed: number;
}

export interface CppPhase34TestResult {
  status: string;
  tests_passed: number;
}

export interface CppBpeTrainResult {
  status: string;
  vocab_size: number;
  merges_count: number;
  output_path: string;
  error?: string;
}

export interface CppQaTrainResult {
  status: string;
  total_steps: number;
  initial_train_loss: number;
  final_train_loss: number;
  initial_val_loss: number;
  final_val_loss: number;
  initial_perplexity: number;
  final_perplexity: number;
  weight_delta_norm: number;
  checkpoint_path: string;
  tokenizer_path: string;
  vocab_size: number;
  sample_generation: string;
  duration_s: number;
  error?: string;
}

export interface CppQaAnswerResult {
  status: string;
  question: string;
  prompt: string;
  answer: string;
  full_text: string;
  prompt_tokens?: number;
  tokens_generated: number;
  max_seq_len?: number;
  vocab_size?: number;
  use_rope?: boolean;
  tie_embeddings?: boolean;
  duration_ms: number;
  tokens_per_second: number;
  finish_reason: string;
  error?: string;
}

export interface CppPhase35TestResult {
  status: string;
  tests_passed: number;
}

export interface CppCapacityConfigEntry {
  name: string;
  params: number;
  dim: number;
  intermediate_dim: number;
  layers: number;
  heads: number;
  context_length: number;
  weights_mb: number;
  est_training_mb: number;
  est_inference_mb: number;
  avg_step_ms: number;
  tokens_per_sec: number;
  peak_rss_mb: number;
  feasible: boolean;
}

export interface CppCapacityBenchmarkResult {
  status: string;
  vocab_size: number;
  configs: CppCapacityConfigEntry[];
}

export interface CppPhase35TrainResult {
  status: string;
  total_parameters: number;
  vocab_size: number;
  dim: number;
  layers: number;
  heads: number;
  seq_len: number;
  examples: number;
  total_tokens: number;
  masked_target_tokens: number;
  unmasked_target_tokens: number;
  total_steps: number;
  initial_train_loss: number;
  final_train_loss: number;
  initial_val_loss: number;
  final_val_loss: number;
  best_val_loss: number;
  best_checkpoint_step: number;
  best_perplexity: number;
  overfitting_detected: boolean;
  tokens_per_sec: number;
  peak_rss_mb: number;
  duration_s: number;
  best_checkpoint_path: string;
  tokenizer_path: string;
  sample_generation: string;
  error?: string;
}

export interface CppUnseenSampleResult {
  category: string;
  expected_behavior: string;
  question: string;
  answer: string;
  tokens_generated: number;
  duration_ms: number;
  repetition_rate: number;
  word_repetition_rate?: number;
  relevance_score?: number;
  coherence_score?: number;
  factual_accuracy_score?: number;
  instruction_following_score?: number;
  clean_stop: boolean;
  relevance_hit: boolean;
}

export interface CppUnseenEvalResult {
  status: string;
  checkpoint: string;
  total_parameters: number;
  disjoint_verified: boolean;
  total_questions: number;
  non_empty_responses: number;
  clean_stop_count: number;
  relevance_hit_count: number;
  uncertainty_questions: number;
  uncertainty_matched: number;
  avg_relevance_score?: number;
  avg_coherence_score?: number;
  avg_factual_accuracy_score?: number;
  avg_instruction_following_score?: number;
  avg_latency_ms: number;
  avg_tokens_per_sec: number;
  avg_repetition_rate: number;
  avg_word_repetition_rate?: number;
  peak_rss_mb: number;
  samples: CppUnseenSampleResult[];
  error?: string;
}

export interface CppPhase36TestResult {
  status: string;
  tests_passed: number;
}

export interface CppDatasetAuditResult {
  status: string;
  valid: boolean;
  dataset_version: string;
  content_hash: string;
  source: string;
  license: string;
  license_verified: boolean;
  documents: number;
  examples: number;
  train_examples: number;
  val_examples: number;
  bytes: number;
  total_tokens: number;
  unique_tokens: number;
  train_tokens: number;
  val_tokens: number;
  masked_tokens: number;
  unmasked_tokens: number;
  exact_duplicates_removed: number;
  near_duplicates_removed: number;
  quality_issues_removed: number;
  contamination_count: number;
  duplication_rate: number;
  quality_gate_passed: boolean;
  error?: string;
}

export interface CppPhase36TrainResult {
  status: string;
  experiment_id: string;
  experiment_path: string;
  dataset_version: string;
  license: string;
  license_verified: boolean;
  total_parameters: number;
  vocab_size: number;
  intermediate_dim?: number;
  max_seq_len?: number;
  use_rope?: boolean;
  tie_embeddings?: boolean;
  documents: number;
  examples: number;
  bytes: number;
  total_tokens: number;
  unique_tokens: number;
  masked_tokens: number;
  unmasked_tokens: number;
  contamination_count: number;
  total_steps: number;
  initial_train_loss: number;
  final_train_loss: number;
  initial_val_loss: number;
  final_val_loss: number;
  best_val_loss: number;
  best_perplexity: number;
  best_checkpoint_step: number;
  overfitting_detected: boolean;
  tokens_per_sec: number;
  peak_rss_mb: number;
  duration_s: number;
  best_checkpoint_path: string;
  tokenizer_path: string;
  eval_evaluated: boolean;
  eval_questions: number;
  eval_relevance_hits: number;
  eval_coherence: number;
  eval_factual: number;
  eval_instruction: number;
  eval_repetition: number;
  error?: string;
}

export class NexaNativeCppError extends Error {
  readonly code: string;
  readonly binaryPath?: string;
  readonly exitCode?: number | null;
  readonly stderr?: string;

  constructor(message: string, code: string, binaryPath?: string, exitCode?: number | null, stderr?: string) {
    super(`[NEXA_CPP_ERROR:${code}] ${message}${stderr ? `\nC++ stderr: ${stderr}` : ''}`);
    this.name = 'NexaNativeCppError';
    this.code = code;
    this.binaryPath = binaryPath;
    this.exitCode = exitCode;
    this.stderr = stderr;
  }
}

export class NexaCppBridge {
  private binaryPath: string | null = null;
  private readonly defaultTimeoutMs: number = 10000;

  constructor(explicitBinaryPath?: string) {
    if (explicitBinaryPath) {
      this.binaryPath = explicitBinaryPath;
    } else {
      this.discoverBinary();
    }
  }

  /**
   * Probes filesystem for compiled C++ executable.
   */
  public discoverBinary(): string | null {
    const candidatePaths = [
      process.env.NEXA_CPP_BINARY_PATH ? path.resolve(process.cwd(), process.env.NEXA_CPP_BINARY_PATH) : '',
      path.resolve(process.cwd(), 'bin', 'nexa-runtime-cli'),
      path.resolve(process.cwd(), 'native', 'build', 'nexa-runtime-cli'),
      '/usr/local/bin/nexa-runtime-cli',
      '/data/data/com.termux/files/usr/bin/nexa-runtime-cli'
    ].filter(Boolean);

    for (const p of candidatePaths) {
      if (fs.existsSync(p)) {
        try {
          fs.accessSync(p, fs.constants.X_OK);
          this.binaryPath = p;
          return p;
        } catch {
          // If file exists but X_OK is missing (e.g. after git restore), attempt chmod
          try {
            fs.chmodSync(p, 0o755);
            fs.accessSync(p, fs.constants.X_OK);
            this.binaryPath = p;
            return p;
          } catch {
            // Exists but cannot be made executable
          }
        }
      }
    }

    this.binaryPath = null;
    return null;
  }

  public getBinaryPath(): string | null {
    if (!this.binaryPath) {
      this.discoverBinary();
    }
    return this.binaryPath;
  }

  public isAvailable(): boolean {
    return this.getBinaryPath() !== null;
  }

  /**
   * Executes C++ binary with safe argument passing (execFile), strict timeout,
   * and structured JSON response parsing.
   */
  private executeCommand<T>(args: string[], timeoutMs: number = this.defaultTimeoutMs): Promise<T> {
    const bin = this.getBinaryPath();
    if (!bin) {
      return Promise.reject(
        new NexaNativeCppError(
          'Compiled C++ executable "nexa-runtime-cli" was not found in PATH or project bin/.\n' +
          'Execute: cd native && cmake -B build && cmake --build build -j',
          'ERR_EXECUTABLE_NOT_FOUND'
        )
      );
    }

    return new Promise((resolve, reject) => {
      // execFile bypasses shell interpolation (immune to shell injection)
      const child = execFile(
        bin,
        args,
        {
          timeout: timeoutMs,
          maxBuffer: 4 * 1024 * 1024, // 4MB output buffer
          killSignal: 'SIGKILL'
        },
        (error, stdout, stderr) => {
          const extractJson = (raw: string): T | null => {
            const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
            for (let i = lines.length - 1; i >= 0; i--) {
              if (lines[i].startsWith('{') && lines[i].endsWith('}')) {
                try {
                  return JSON.parse(lines[i]) as T;
                } catch {
                  // try fallback
                }
              }
            }
            const firstBrace = raw.indexOf('{');
            const lastBrace = raw.lastIndexOf('}');
            if (firstBrace !== -1 && lastBrace !== -1 && lastBrace >= firstBrace) {
              try {
                return JSON.parse(raw.substring(firstBrace, lastBrace + 1)) as T;
              } catch {
                return null;
              }
            }
            return null;
          };

          if (error) {
            // If the command produced a valid JSON payload on stdout despite exit code, parse it
            if (stdout && stdout.trim()) {
              const parsed = extractJson(stdout.trim());
              if (parsed) {
                return resolve(parsed);
              }
            }

            if ((error as any).killed) {
              return reject(
                new NexaNativeCppError(
                  `C++ process timed out after ${timeoutMs}ms and was killed with SIGKILL`,
                  'ERR_PROCESS_TIMEOUT',
                  bin,
                  null,
                  stderr
                )
              );
            }

            if ((error as any).code === 'ENOENT') {
              return reject(
                new NexaNativeCppError(
                  `C++ executable not found at "${bin}": ${error.message}`,
                  'ERR_EXECUTABLE_NOT_FOUND',
                  bin,
                  null,
                  stderr
                )
              );
            }

            return reject(
              new NexaNativeCppError(
                `C++ executable failed with exit code ${(error as any).code || 'unknown'}: ${error.message}`,
                'ERR_PROCESS_EXECUTION_FAILED',
                bin,
                (error as any).code,
                stderr || stdout
              )
            );
          }

          const trimmed = stdout.trim();
          if (!trimmed) {
            return reject(
              new NexaNativeCppError(
                'C++ process returned empty output',
                'ERR_EMPTY_OUTPUT',
                bin,
                0,
                stderr
              )
            );
          }

          try {
            const parsed = extractJson(trimmed);
            if (!parsed) {
              throw new Error('No valid JSON object found in stdout');
            }
            resolve(parsed);
          } catch (parseErr: any) {
            reject(
              new NexaNativeCppError(
                `Failed to parse JSON output from C++ process: ${parseErr.message}\nRaw output: ${trimmed}`,
                'ERR_JSON_PARSE_FAILED',
                bin,
                0,
                stderr
              )
            );
          }
        }
      );

      // Handle child process stream errors
      child.on('error', (err: any) => {
        if (err.code === 'ENOENT') {
          reject(
            new NexaNativeCppError(
              `C++ executable not found at "${bin}": ${err.message}`,
              'ERR_EXECUTABLE_ENOENT',
              bin
            )
          );
        } else {
          reject(
            new NexaNativeCppError(
              `Failed to spawn C++ child process: ${err.message}`,
              'ERR_SPAWN_FAILED',
              bin
            )
          );
        }
      });
    });
  }

  /**
   * Retrieves C++ runtime engine info and architecture telemetry.
   */
  public async getRuntimeInfo(): Promise<CppRuntimeInfo> {
    const bin = this.getBinaryPath();
    const res = await this.executeCommand<CppRuntimeInfo>(['--info-json']);
    return {
      ...res,
      binaryPath: bin || ''
    };
  }

  /**
   * Executes matrix multiplication and neural network math verification in C++.
   */
  public async runTensorTests(): Promise<CppTensorTestResult> {
    return this.executeCommand<CppTensorTestResult>(['--test-tensor-json']);
  }

  /**
   * Executes transformer forward pass in C++.
   */
  public async runForwardPass(dim: number = 128, seqLen: number = 8): Promise<CppForwardPassResult> {
    return this.executeCommand<CppForwardPassResult>([
      '--forward-pass-json',
      String(Math.floor(dim)),
      String(Math.floor(seqLen))
    ]);
  }

  /**
   * Encodes text into tokens using C++ NexaBpeTokenizer (if tokenizerPath supplied) or NexaByteTokenizer.
   */
  public async tokenize(text: string, tokenizerPath?: string): Promise<number[]> {
    const args = ['--tokenize-json', text];
    if (tokenizerPath) {
      const resolvedTok = path.isAbsolute(tokenizerPath)
        ? tokenizerPath
        : path.resolve(process.cwd(), tokenizerPath);
      args.push(resolvedTok);
    }
    const res = await this.executeCommand<CppTokenizeResult>(args);
    if (res.status !== 'ok' || !Array.isArray(res.tokens)) {
      throw new NexaNativeCppError(
        res.error || 'C++ tokenizer returned unexpected payload',
        'ERR_TOKENIZER_INVALID_PAYLOAD'
      );
    }
    return res.tokens;
  }

  /**
   * Inspects .nexa model file header in C++.
   */
  public async inspectModel(filePath: string): Promise<CppInspectResult> {
    const resolved = path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);
    return this.executeCommand<CppInspectResult>(['--inspect-json', resolved]);
  }

  /**
   * Runs C++ autograd & AdamW unit tests natively.
   */
  public async runTrainingTests(): Promise<CppTrainingTestResult> {
    return this.executeCommand<CppTrainingTestResult>(['--test-training-json']);
  }

  /**
   * Trains NEXA-1 Nano on synthetic data using native C++ autograd and AdamW.
   */
  public async trainNano(steps: number = 40, lr: number = 0.003): Promise<CppTrainingRunResult> {
    return this.executeCommand<CppTrainingRunResult>(
      ['--train-nano-json', String(steps), String(lr)],
      30000
    );
  }

  /**
   * Validates .nexa checkpoint structure, architecture, and weights.
   */
  public async validateCheckpoint(checkpointPath: string): Promise<CppCheckpointValidationResult> {
    const resolved = path.isAbsolute(checkpointPath) ? checkpointPath : path.resolve(process.cwd(), checkpointPath);
    return this.executeCommand<CppCheckpointValidationResult>(['--validate-checkpoint-json', resolved]);
  }

  /**
   * Generates text autoregressively using a trained .nexa model checkpoint.
   */
  public async generateText(
    checkpointPath: string,
    prompt: string,
    options?: {
      maxTokens?: number;
      temperature?: number;
      topK?: number;
      topP?: number;
      repetitionPenalty?: number;
      seed?: number;
      tokenizerPath?: string;
    }
  ): Promise<CppGenerationResult> {
    const resolved = path.isAbsolute(checkpointPath) ? checkpointPath : path.resolve(process.cwd(), checkpointPath);
    const maxTokens = options?.maxTokens ?? 32;
    const temp = options?.temperature ?? 0.7;
    const topK = options?.topK ?? 40;
    const topP = options?.topP ?? 0.9;
    const repPenalty = options?.repetitionPenalty ?? 1.15;
    const seed = options?.seed ?? 42;

    const args = [
      '--generate-json',
      resolved,
      prompt,
      String(maxTokens),
      String(temp),
      String(topK),
      String(topP),
      String(repPenalty),
      String(seed)
    ];

    if (options?.tokenizerPath) {
      const resolvedTok = path.isAbsolute(options.tokenizerPath)
        ? options.tokenizerPath
        : path.resolve(process.cwd(), options.tokenizerPath);
      args.push(resolvedTok);
    }

    return this.executeCommand<CppGenerationResult>(args, 30000);
  }

  /**
   * Trains NEXA-1 Nano using a plain-text dataset with UTF-8 validation and train/val split.
   */
  public async trainDatasetPipeline(
    datasetPath: string,
    options?: {
      steps?: number;
      lr?: number;
      seqLen?: number;
      split?: number;
      checkpointPath?: string;
    }
  ): Promise<CppDatasetTrainingResult> {
    const resolvedDs = path.isAbsolute(datasetPath) ? datasetPath : path.resolve(process.cwd(), datasetPath);
    const steps = options?.steps ?? 50;
    const lr = options?.lr ?? 0.003;
    const seqLen = options?.seqLen ?? 16;
    const split = options?.split ?? 0.8;
    const ckpt = options?.checkpointPath
      ? (path.isAbsolute(options.checkpointPath) ? options.checkpointPath : path.resolve(process.cwd(), options.checkpointPath))
      : path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_nano_trained.nexa');

    return this.executeCommand<CppDatasetTrainingResult>(
      ['--train-dataset-json', resolvedDs, String(steps), String(lr), String(seqLen), String(split), ckpt],
      45000
    );
  }

  /**
   * Runs Phase 3.3 generation, sampling, checkpoint validation, and evaluation unit tests.
   */
  public async runGenerationTests(): Promise<CppGenerationTestResult> {
    return this.executeCommand<CppGenerationTestResult>(['--test-generation-json']);
  }

  /**
   * Runs Phase 3.4 language understanding tests (BPE, multi-file dataset, LR warmup/decay, gradient scaling).
   */
  public async runPhase34Tests(): Promise<CppPhase34TestResult> {
    return this.executeCommand<CppPhase34TestResult>(['--test-phase34-json']);
  }

  /**
   * Trains a native BPE tokenizer on a corpus file or directory and writes a .nbt file.
   */
  public async trainBpe(
    corpusPath: string,
    outputPath: string,
    targetVocab: number = 280,
    minFreq: number = 2
  ): Promise<CppBpeTrainResult> {
    const resolvedCorpus = path.isAbsolute(corpusPath) ? corpusPath : path.resolve(process.cwd(), corpusPath);
    const resolvedOut = path.isAbsolute(outputPath) ? outputPath : path.resolve(process.cwd(), outputPath);
    return this.executeCommand<CppBpeTrainResult>(
      ['--train-bpe-json', resolvedCorpus, resolvedOut, String(targetVocab), String(minFreq)],
      30000
    );
  }

  /**
   * Trains NEXA-1 Nano natively on a QA dataset directory using BPE tokenization,
   * mini-batch gradient accumulation, and warmup + cosine LR decay.
   */
  public async trainQa(options?: {
    datasetDir?: string;
    checkpointPath?: string;
    tokenizerPath?: string;
    steps?: number;
    batchSize?: number;
    gradAccum?: number;
    lr?: number;
    warmupSteps?: number;
    seqLen?: number;
  }): Promise<CppQaTrainResult> {
    const dsDir = options?.datasetDir
      ? (path.isAbsolute(options.datasetDir) ? options.datasetDir : path.resolve(process.cwd(), options.datasetDir))
      : path.resolve(process.cwd(), 'data', 'datasets', 'qa');
    const ckpt = options?.checkpointPath
      ? (path.isAbsolute(options.checkpointPath) ? options.checkpointPath : path.resolve(process.cwd(), options.checkpointPath))
      : path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_nano_qa.nexa');
    const nbt = options?.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath) ? options.tokenizerPath : path.resolve(process.cwd(), options.tokenizerPath))
      : path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_qa.nbt');

    return this.executeCommand<CppQaTrainResult>(
      [
        '--train-qa-json',
        dsDir,
        ckpt,
        nbt,
        String(options?.steps ?? 60),
        String(options?.batchSize ?? 2),
        String(options?.gradAccum ?? 1),
        String(options?.lr ?? 0.003),
        String(options?.warmupSteps ?? 10),
        String(options?.seqLen ?? 32)
      ],
      120000
    );
  }

  /**
   * Generates a natural-language answer using a trained NEXA-1 Nano QA checkpoint and BPE tokenizer.
   */
  public async answerQuestion(
    checkpointPath: string,
    question: string,
    options?: {
      tokenizerPath?: string;
      maxTokens?: number;
      temperature?: number;
      seed?: number;
    }
  ): Promise<CppQaAnswerResult> {
    const ckpt = path.isAbsolute(checkpointPath) ? checkpointPath : path.resolve(process.cwd(), checkpointPath);
    const nbt = options?.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath) ? options.tokenizerPath : path.resolve(process.cwd(), options.tokenizerPath))
      : path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_qa.nbt');

    return this.executeCommand<CppQaAnswerResult>(
      [
        '--answer-json',
        ckpt,
        question,
        nbt,
        String(options?.maxTokens ?? 48),
        String(options?.temperature ?? 0.5),
        String(options?.seed ?? 42)
      ],
      30000
    );
  }

  /**
   * Runs Phase 3.5 verification tests (numerical gradients, checkpoint recovery,
   * prompt masking, unseen test disjoint check, best val checkpoint selection).
   */
  public async runPhase35Tests(): Promise<CppPhase35TestResult> {
    return this.executeCommand<CppPhase35TestResult>(['--test-phase35-json'], 60000);
  }

  /**
   * Benchmarks realistic Transformer configurations on CPU for capacity & RAM planning.
   */
  public async benchmarkConfigs(vocabSize: number = 768): Promise<CppCapacityBenchmarkResult> {
    return this.executeCommand<CppCapacityBenchmarkResult>(
      ['--benchmark-configs-json', String(vocabSize)],
      60000
    );
  }

  /**
   * Trains a Phase 3.5 instruction-tuned model on the expanded multi-domain corpus.
   */
  public async trainPhase35(options?: {
    datasetDir?: string;
    bestCheckpointPath?: string;
    tokenizerPath?: string;
    steps?: number;
    dim?: number;
    layers?: number;
    heads?: number;
    vocabSize?: number;
    seqLen?: number;
    batchSize?: number;
    gradAccum?: number;
    lr?: number;
    seed?: number;
  }): Promise<CppPhase35TrainResult> {
    const dsDir = options?.datasetDir
      ? (path.isAbsolute(options.datasetDir) ? options.datasetDir : path.resolve(process.cwd(), options.datasetDir))
      : path.resolve(process.cwd(), 'data', 'datasets', 'phase35', 'train_val');
    const ckpt = options?.bestCheckpointPath
      ? (path.isAbsolute(options.bestCheckpointPath) ? options.bestCheckpointPath : path.resolve(process.cwd(), options.bestCheckpointPath))
      : path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase35_best.nexa');
    const nbt = options?.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath) ? options.tokenizerPath : path.resolve(process.cwd(), options.tokenizerPath))
      : path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase35.nbt');

    return this.executeCommand<CppPhase35TrainResult>(
      [
        '--train-phase35-json',
        dsDir,
        ckpt,
        nbt,
        String(options?.steps ?? 250),
        String(options?.dim ?? 128),
        String(options?.layers ?? 4),
        String(options?.heads ?? 4),
        String(options?.vocabSize ?? 768),
        String(options?.seqLen ?? 64),
        String(options?.batchSize ?? 2),
        String(options?.gradAccum ?? 2),
        String(options?.lr ?? 0.0025),
        String(options?.seed ?? 42)
      ],
      300000
    );
  }

  /**
   * Evaluates a trained checkpoint across all 50+ unseen test questions.
   */
  public async evaluateUnseen(options?: {
    checkpointPath?: string;
    tokenizerPath?: string;
    unseenQuestionsPath?: string;
    trainValDir?: string;
    maxTokens?: number;
    temperature?: number;
    seed?: number;
  }): Promise<CppUnseenEvalResult> {
    const ckpt = options?.checkpointPath
      ? (path.isAbsolute(options.checkpointPath) ? options.checkpointPath : path.resolve(process.cwd(), options.checkpointPath))
      : path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase35_best.nexa');
    const nbt = options?.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath) ? options.tokenizerPath : path.resolve(process.cwd(), options.tokenizerPath))
      : path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase35.nbt');
    const unseen = options?.unseenQuestionsPath
      ? (path.isAbsolute(options.unseenQuestionsPath) ? options.unseenQuestionsPath : path.resolve(process.cwd(), options.unseenQuestionsPath))
      : path.resolve(process.cwd(), 'data', 'datasets', 'phase35', 'unseen_test_qa.txt');
    const trainDir = options?.trainValDir
      ? (path.isAbsolute(options.trainValDir) ? options.trainValDir : path.resolve(process.cwd(), options.trainValDir))
      : path.resolve(process.cwd(), 'data', 'datasets', 'phase35', 'train_val');

    return this.executeCommand<CppUnseenEvalResult>(
      [
        '--eval-unseen-json',
        ckpt,
        nbt,
        unseen,
        trainDir,
        String(options?.maxTokens ?? 48),
        String(options?.temperature ?? 0.4),
        String(options?.seed ?? 42)
      ],
      180000
    );
  }

  /**
   * Runs Phase 3.6 native unit tests (provenance, unverified license rejection,
   * 8-way quality filtering, contamination detection, multi-turn masking, ExperimentManager).
   */
  public async runPhase36Tests(): Promise<CppPhase36TestResult> {
    return this.executeCommand<CppPhase36TestResult>(['--test-phase36-json'], 60000);
  }

  /**
   * Audits a managed dataset directory or manifest for quality, provenance,
   * licensing verification, token stats, and evaluation contamination.
   */
  public async auditDataset(options: {
    datasetPath: string;
    unseenQuestionsPath?: string;
    tokenizerPath?: string;
    vocabSize?: number;
    requireVerifiedLicense?: boolean;
  }): Promise<CppDatasetAuditResult> {
    const ds = path.isAbsolute(options.datasetPath)
      ? options.datasetPath
      : path.resolve(process.cwd(), options.datasetPath);
    const unseen = options.unseenQuestionsPath
      ? (path.isAbsolute(options.unseenQuestionsPath)
        ? options.unseenQuestionsPath
        : path.resolve(process.cwd(), options.unseenQuestionsPath))
      : '';
    const nbt = options.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath)
        ? options.tokenizerPath
        : path.resolve(process.cwd(), options.tokenizerPath))
      : '';

    return this.executeCommand<CppDatasetAuditResult>(
      [
        '--audit-dataset-json',
        ds,
        unseen,
        nbt,
        String(options.vocabSize ?? 768),
        options.requireVerifiedLicense ? '1' : '0'
      ],
      60000
    );
  }

  /**
   * Runs reproducible Phase 3.6 instruction training and persists ExperimentManager JSON logs.
   */
  public async trainPhase36(options: {
    datasetPath: string;
    bestCheckpointPath?: string;
    tokenizerPath?: string;
    unseenQuestionsPath?: string;
    experimentId?: string;
    steps?: number;
    dim?: number;
    layers?: number;
    heads?: number;
    vocabSize?: number;
    seqLen?: number;
    batchSize?: number;
    gradAccum?: number;
    lr?: number;
    earlyStopPatience?: number;
    seed?: number;
    requireVerifiedLicense?: boolean;
    useRope?: boolean;
    tieEmbeddings?: boolean;
    intermediateDim?: number;
    resumeCheckpointPath?: string;
    minLr?: number;
  }): Promise<CppPhase36TrainResult> {
    const ds = path.isAbsolute(options.datasetPath)
      ? options.datasetPath
      : path.resolve(process.cwd(), options.datasetPath);
    const ckpt = options.bestCheckpointPath
      ? (path.isAbsolute(options.bestCheckpointPath)
        ? options.bestCheckpointPath
        : path.resolve(process.cwd(), options.bestCheckpointPath))
      : path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase36_best.nexa');
    const nbt = options.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath)
        ? options.tokenizerPath
        : path.resolve(process.cwd(), options.tokenizerPath))
      : path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase36.nbt');
    const unseen = options.unseenQuestionsPath
      ? (path.isAbsolute(options.unseenQuestionsPath)
        ? options.unseenQuestionsPath
        : path.resolve(process.cwd(), options.unseenQuestionsPath))
      : '';
    const resumeCkpt = options.resumeCheckpointPath
      ? (path.isAbsolute(options.resumeCheckpointPath)
        ? options.resumeCheckpointPath
        : path.resolve(process.cwd(), options.resumeCheckpointPath))
      : '';

    return this.executeCommand<CppPhase36TrainResult>(
      [
        '--train-phase36-json',
        ds,
        ckpt,
        nbt,
        unseen,
        options.experimentId ?? '',
        String(options.steps ?? 160),
        String(options.dim ?? 64),
        String(options.layers ?? 2),
        String(options.heads ?? 2),
        String(options.vocabSize ?? 768),
        String(options.seqLen ?? 64),
        String(options.batchSize ?? 2),
        String(options.gradAccum ?? 2),
        String(options.lr ?? 0.0025),
        String(options.earlyStopPatience ?? 6),
        String(options.seed ?? 42),
        options.requireVerifiedLicense ? '1' : '0',
        options.useRope ? '1' : '0',
        options.tieEmbeddings ? '1' : '0',
        String(options.intermediateDim ?? 0),
        resumeCkpt,
        String(options.minLr ?? 0.0001)
      ],
      900000
    );
  }

  /**
   * Evaluates a trained checkpoint with Phase 3.6 multi-metric scoring.
   */
  public async evaluateUnseenV2(options?: {
    checkpointPath?: string;
    tokenizerPath?: string;
    unseenQuestionsPath?: string;
    trainValDir?: string;
    maxTokens?: number;
    temperature?: number;
    seed?: number;
  }): Promise<CppUnseenEvalResult> {
    const ckpt = options?.checkpointPath
      ? (path.isAbsolute(options.checkpointPath) ? options.checkpointPath : path.resolve(process.cwd(), options.checkpointPath))
      : path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase36_best.nexa');
    const nbt = options?.tokenizerPath
      ? (path.isAbsolute(options.tokenizerPath) ? options.tokenizerPath : path.resolve(process.cwd(), options.tokenizerPath))
      : path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase36.nbt');
    const unseen = options?.unseenQuestionsPath
      ? (path.isAbsolute(options.unseenQuestionsPath) ? options.unseenQuestionsPath : path.resolve(process.cwd(), options.unseenQuestionsPath))
      : path.resolve(process.cwd(), 'data', 'datasets', 'phase36', 'unseen_test_expanded.txt');
    const trainDir = options?.trainValDir
      ? (path.isAbsolute(options.trainValDir) ? options.trainValDir : path.resolve(process.cwd(), options.trainValDir))
      : path.resolve(process.cwd(), 'data', 'datasets', 'phase36', 'tier3_large', 'train_val');

    return this.executeCommand<CppUnseenEvalResult>(
      [
        '--eval-unseen-v2-json',
        ckpt,
        nbt,
        unseen,
        trainDir,
        String(options?.maxTokens ?? 48),
        String(options?.temperature ?? 0.4),
        String(options?.seed ?? 42)
      ],
      240000
    );
  }
}
