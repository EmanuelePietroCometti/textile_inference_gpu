#pragma once
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <vector>
#include "PipelineConfig.h"
#include "ContractMetadata.h"

/**
 * @brief RAII wrapper for an ONNX Runtime (ORT) session and its dedicated GPU context.
 *
 * @details This class encapsulates the ORT session, `IoBinding`, CUDA streams, hardware timing events,
 * and device-side memory buffers. It is strictly designed for a single-thread execution model:
 * nothing is shared between threads, and `RunBatch` is NOT thread-safe.
 *
 * To guarantee zero-allocation on the critical path, `Ort::Value` tensors are constructed exactly once
 * during initialization and repeatedly bound to the execution context. Device pointers and tensor
 * shapes remain fixed throughout the session's lifetime. If `LoopBatch1` is enabled (running a batch
 * sequentially on a batch-1 model), this prevents the costly reallocation of `3 * batchSize` tensors
 * during every inference cycle.
 */
class OrtSessionConfig {
public:
    /**
     * @brief Constructs the session, configures CUDA providers, and allocates device memory.
     *
     * @param env The global ONNX Runtime environment instance.
     * @param cfg The pipeline configuration dictating execution providers (CUDA/TensorRT) and batch settings.
     * @throws std::runtime_error if model loading, memory allocation, or tensor binding fails.
     */
    OrtSessionConfig(Ort::Env& env, const PipelineConfig& cfg);

    /**
     * @brief Destructor. Safely cleans up CUDA streams, events, and device buffers.
     */
    ~OrtSessionConfig();

    // Disable copy semantics to ensure unique ownership of GPU resources.
    OrtSessionConfig(const OrtSessionConfig&) = delete;
    OrtSessionConfig& operator=(const OrtSessionConfig&) = delete;

    // --- Geometry Accessors ---
    // The following methods retrieve the geometric dimensions read directly from the ONNX graph.

    /** @brief Retrieves the fixed batch size configured for this session. */
    int BatchSize() const;

    /** @brief Retrieves the number of expected color channels (e.g., 3 for RGB). */
    int ModelChannels() const;

    /** @brief Retrieves the expected input image height. */
    int ModelHeight() const;

    /** @brief Retrieves the expected input image width. */
    int ModelWidth() const;

    /** @brief Retrieves the total number of elements in the input tensor (Batch * Channels * Height * Width). */
    std::size_t InputElems() const;

    /** @brief Retrieves the total number of elements in the classification score output tensor (Batch). */
    std::size_t ScoreElems() const;

    /** @brief Retrieves the total number of elements in the segmentation map output tensor (Batch * mapHeight * mapWidth). */
    std::size_t MapElems() const;

    /** @brief Retrieves the height of the output segmentation map. */
    int MapHeight() const;

    /** @brief Retrieves the width of the output segmentation map. */
    int MapWidth() const;

    /**
     * @brief Retrieves the embedded ONNX contract metadata parsed during session initialization.
     * @return A constant reference to the extracted `ContractMetadata`.
     */
    const ContractMetadata& Contract() const;

    /**
     * @brief Container for hardware-accurate execution timings.
     */
    struct Timings {
        double h2dMs = 0.0; ///< Host-to-Device memory transfer time in milliseconds.
        double runMs = 0.0; ///< GPU kernel execution time in milliseconds.
        double d2hMs = 0.0; ///< Device-to-Host memory transfer time in milliseconds.
    };

    /**
     * @brief Executes a complete, blocking inference batch synchronously.
     *
     * @details Orchestrates the entire inference lifecycle using ORT `IoBinding` on the dedicated
     * CUDA stream. The pipeline includes:
     * 1. Host-to-Device (H2D) transfer of input pixels.
     * 2. ONNX/TensorRT kernel execution.
     * 3. Device-to-Host (D2H) transfer of scores and masks.
     * 4. Full stream synchronization.
     *
     * Upon returning, all output buffers are fully populated and safe to read.
     *
     * @param input Pointer to the contiguous pinned host buffer holding the input data [B*C*H*W].
     * @param scores Pointer to the pinned host buffer where classification scores will be written [B].
     * @param map Pointer to the pinned host buffer where segmentation masks will be written [B*mH*mW].
     * @param t Reference to a `Timings` struct that will be populated with CUDA Event-based durations.
     */
    void RunBatch(const float* input, float* scores, float* map, Timings& t);

    /**
     * @brief Executes dummy inference passes to trigger JIT compilation and warm up GPU clocks.
     *
     * @details Critical for TensorRT execution providers, as engine building and memory optimization
     * occur on the first inference pass. Warming up ensures real-time latency stability for the first real frame.
     *
     * @param runs The number of dummy iterations to execute.
     */
    void Warmup(int runs);

private:
    /**
     * @brief Parses node shapes from the ONNX graph and applies dynamic batch settings.
     *
     * @details Extracts spatial dimensions (C, H, W) directly from the model signatures.
     * If the model supports dynamic batching (e.g., `-1` or `?` for the batch dimension),
     * or if a batch override is required, it dynamically sets `batch_` based on the
     * requested `PipelineConfig`.
     *
     * @param cfg The pipeline configuration providing the target batch size override.
     */
    void ReadModelGeometry(const PipelineConfig& cfg);

    /**
     * @brief Initializes the CUDA hardware timing events.
     *
     * @details Instantiates `evStart_`, `evH2D_`, `evRun_`, and `evD2H_` using `cudaEventCreate`.
     * These events are injected into the CUDA stream to measure precise asynchronous hardware
     * latency for each stage without blocking the CPU.
     */
    void CreateEvents();

    /** @brief Allocates raw memory on the GPU using `cudaMallocAsync` on the session's stream. */
    void AllocateDeviceBuffers();

    /** @brief Constructs the `Ort::Value` tensor wrappers and binds them to the device pointers. */
    void BuildTensors();

    /**
     * @brief Safely releases all allocated GPU resources and OS handles.
     *
     * @details Centralized teardown logic invoked by the destructor or during initialization
     * failures. Frees device memory (`cudaFree`), destroys CUDA events (`cudaEventDestroy`),
     * and ensures no memory leaks occur if the session construction aborts early.
     */
    void Cleanup();

    /**
     * @brief Triggers the ONNX Runtime execution using the pre-bound tensors.
     *
     * @details Invokes `session_.Run()` with the configured `IoBinding`. Since device pointers
     * and memory spaces are already locked in during `BuildTensors()`, this call strictly
     * dispatches the compute kernels to the CUDA stream with zero CPU allocation overhead.
     */
    void RunAllBound();

    Ort::Session session_;         ///< The core ONNX Runtime session.
    Ort::IoBinding binding_;       ///< Handles optimal memory transfers and static device pointer binding.
    Ort::MemoryInfo deviceMemInfo_;///< Memory allocation specification for the execution provider.
    ContractMetadata contract_;    ///< Custom metadata extracted from the ONNX model.

    cudaStream_t stream_ = nullptr; ///< Dedicated CUDA stream for asynchronous, non-blocking GPU dispatching.
    cudaEvent_t evStart_ = nullptr, evH2D_ = nullptr, evRun_ = nullptr, evD2H_ = nullptr; ///< Hardware timers.

    float* dInput_ = nullptr;       ///< Device pointer for the input tensor.
    float* dScore_ = nullptr;       ///< Device pointer for the classification score tensor.
    float* dMap_ = nullptr;         ///< Device pointer for the segmentation map tensor.

    std::vector<Ort::Value> inputTensors_; ///< Pre-constructed zero-copy tensor headers for inputs.
    std::vector<Ort::Value> scoreTensors_; ///< Pre-constructed zero-copy tensor headers for scores.
    std::vector<Ort::Value> mapTensors_;   ///< Pre-constructed zero-copy tensor headers for maps.

    std::string inputName_, scoreName_, mapName_;
    int batch_ = 0, modelC_ = 0, modelH_ = 0, modelW_ = 0, mapH_ = 0, mapW_ = 0;
    bool loopBatch1_ = false; ///< Flag indicating if a batch > 1 is processed sequentially through a batch=1 model.
};