#pragma once

#include <string>
#include <cstdint>
#include "IniConfig.h"

/**
 * @brief Defines the available hardware execution providers for the inference engine.
 */
enum class ExecutionProvider : uint8_t {
    CPU = 0,        ///< Fallback execution on standard CPU.
    CUDA = 1,       ///< GPU acceleration using NVIDIA CUDA (FP32).
    TensorRT = 2    ///< Maximum GPU acceleration using NVIDIA TensorRT (supports FP16/INT8).
};

/**
 * @brief Immutable configuration container for the ONNX Runtime inference pipeline.
 *
 * @details Encapsulates all static parameters required to initialize an `OrtSession`.
 * The class guarantees immutability post-creation, making it inherently thread-safe
 * to share across multiple producer/consumer threads.
 */
class PipelineConfig {
public:
    /**
     * @brief Factory method to instantiate and populate a PipelineConfig from an INI file.
     *
     * @param ini A reference to a successfully loaded IniConfig object.
     * @return A fully populated and immutable PipelineConfig instance.
     */
    static PipelineConfig LoadFromIni(const IniConfig& ini);

    /**
     * @brief Validates the logical consistency of the configuration.
     *
     * @param error Reference to a string that will be populated with a detailed
     * description of the first violated constraint (including the offending values).
     * @return `true` if the configuration is fully valid and safe to use, `false` otherwise.
     */
    [[nodiscard]] bool Validate(std::string& error) const;

    /**
     * @brief Generates a comprehensive, single-line summary of all resolved configuration values.
     *
     * @details This is critical for field diagnostics. Logging this string immediately
     * upon startup guarantees that any anomalous runtime behavior can be traced back
     * to the exact initialization state of the deployment environment.
     *
     * @return A formatted diagnostic string.
     */
    std::string Describe() const;

    // --- Getters ---

    /** @brief Retrieves the filesystem path to the ONNX model. */
    const std::wstring& ModelPath() const;

    /** @brief Retrieves the batch size. */
    int BatchSize() const;

    /** @brief Checks if batch=1 loop mapping is enabled. */
    bool LoopBatch1() const;

    /** @brief Retrieves the execution provider (CPU/CUDA/TensorRT). */
    ExecutionProvider Provider() const;

    /** @brief Retrieves the target GPU device ID. */
    int DeviceId() const;

    /** @brief Checks if FP16 precision is enabled for TensorRT. */
    bool EnableFp16() const;

    /** @brief Retrieves the maximum workspace size (in bytes) allowed for TensorRT. */
    std::size_t TrtWorkspaceSize() const;

    /**
     * @brief Retrieves the directory path for the TensorRT engine cache.
     * @details Storing the serialized `*.engine` files here allows multiple sessions
     * or subsequent application launches to bypass the costly JIT engine building phase.
     */
    const std::string& TrtEngineCachePath() const;

    /**
     * @brief Retrieves the directory path for the TensorRT timing cache.
     * @details Reuses kernel profiling data from previous runs to significantly speed up
     * the TensorRT engine builder if the model or batch size changes.
     */
    const std::string& TrtTimingCachePath() const;

    /**
     * @brief Retrieves the TensorRT builder optimization level (0 to 5).
     * @details Higher levels (e.g., 5) yield faster inference at the cost of vastly increased engine build times.
     */
    int TrtBuilderOptLevel() const;

    /**
     * @brief Retrieves the number of dummy inference iterations to execute at startup.
     * @details Critical for triggering lazy-loaded JIT compilation and waking up GPU hardware
     * clocks from their idle power state before the first real frame arrives.
     */
    int WarmupRuns() const;

private:
    /**
     * @brief Private constructor to enforce creation only via the static factory method.
     */
    PipelineConfig(std::wstring modelPath, int batchSize, bool loopBatch1,
        ExecutionProvider provider, int deviceId, bool enableFp16,
        std::size_t trtWorkspaceSize, std::string engineCachePath,
        std::string timingCachePath, int builderOptLevel, int warmupRuns);

    // Private encapsulated state
    std::wstring modelPath_;
    int batchSize_;
    bool loopBatch1_;
    ExecutionProvider provider_;
    int deviceId_;
    bool enableFp16_;
    std::size_t trtWorkspaceSize_;
    std::string engineCachePath_;
    std::string timingCachePath_;
    int builderOptLevel_;
    int warmupRuns_;
};