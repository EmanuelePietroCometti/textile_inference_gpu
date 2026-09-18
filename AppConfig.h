#pragma once

#include "PipelineConfig.h"
#include "PatchLayout.h"
#include <string>

/**
 * @brief Master configuration container for the entire application lifecycle.
 *
 * @details This class acts as the single source of truth for all runtime parameters.
 * It aggregates the AI inference settings (`PipelineConfig`), image extraction geometries
 * (`PatchGeometry`), and application-level tuning such as thread counts, queue capacities,
 * Windows OS real-time scheduling, and telemetry.
 *
 * It is strictly immutable post-creation and enforces cross-component constraints
 * to guarantee that the multi-threaded pipeline cannot deadlock due to starvation.
 */
class AppConfig {
public:
    /**
     * @brief Factory method to parse the INI file and build the complete configuration tree.
     *
     * @param iniPath The filesystem path to the main configuration INI file.
     * @return A fully populated, validated, and immutable `AppConfig` instance.
     * @throws std::runtime_error if parsing fails or if cross-constraints are violated.
     */
    static AppConfig LoadFromIni(const std::wstring& iniPath);

    /**
     * @brief Generates a comprehensive, single-line diagnostic summary of the entire configuration.
     * @return A formatted string suitable for logging at application startup.
     */
    std::string Describe() const;

    /** @brief Retrieves the immutable ONNX model configuration. */
    const PipelineConfig& Model() const { return model_; }

    /** @brief Retrieves the immutable patch extraction geometry. */
    const PatchGeometry& Geometry() const { return geometry_; }

    // [Pipeline]

    /** @brief Number of raw frames to pre-allocate in the IPC/Camera buffer pool. */
    std::uint32_t RawFrames() const { return rawFrames_; }

    /** @brief Total number of `PipelineSlot` instances allocated in the main memory pool. */
    std::uint32_t Slots() const { return slots_; }

    /** @brief Maximum capacity of the raw frame intake queue. */
    std::uint32_t QRawCapacity() const { return qRawCap_; }

    /** @brief Maximum capacity of the preprocessing queue (slots ready for inference). */
    std::uint32_t QPrepCapacity() const { return qPrepCap_; }

    /** @brief Maximum capacity of the inference queue (slots ready for post-processing). */
    std::uint32_t QInfCapacity() const { return qInfCap_; }

    /** @brief Number of concurrent threads executing the `PrepStage`. */
    std::uint32_t PrepThreads() const { return prepThreads_; }

    /** @brief Number of concurrent threads executing the `InferStage` (requires 1 GPU session per thread). */
    std::uint32_t InferenceThreads() const { return inferThreads_; }

    /** @brief Number of concurrent threads executing the `PostStage`. */
    std::uint32_t PostThreads() const { return postThreads_; }

    /**
     * @brief Number of internal threads OpenCV is allowed to spawn.
     * @details Used to invoke `cv::setNumThreads()` globally. Set to 1 to prevent OpenCV from
     * thrashing the CPU when you already have a highly parallelized pipeline.
     */
    std::uint32_t OpenCvThreads() const { return openCvThreads_; }

    /** @brief Milliseconds to wait before timing out a read operation on the IPC MMF. */
    std::uint32_t ReadTimeoutMs() const { return readTimeoutMs_; }

    // [Output]

    /** @brief If true, the output map matches the model's native resolution. If false, it matches the input geometry. */
    bool MapAtModelResolution() const { return mapAtModelRes_; }

    /** @brief If true, the post-processing stage allocates and generates binary threshold masks. */
    bool DrawMask() const { return drawMask_; }

    // [RealTime]
    // These flags control Windows API scheduling priorities to minimize jitter.

    /** @brief If true, elevates the entire process priority class (e.g., HIGH_PRIORITY_CLASS). */
    bool ElevateProcess() const { return elevateProcess_; }

    /** @brief If true, elevates PrepStage threads to Time Critical priority. */
    bool PrepRealtime() const { return prepRealtime_; }

    /** @brief If true, elevates InferStage threads to Time Critical priority. */
    bool InferenceRealtime() const { return inferRealtime_; }

    /** @brief If true, elevates PostStage threads to Time Critical priority. */
    bool PostRealtime() const { return postRealtime_; }

    // [Logger]

    /** @brief Maximum character length for a single async log message before truncation. */
    std::uint32_t LogMaxMessageChars() const { return logMaxChars_; }

    /** @brief Maximum number of pending messages in the async logger's lock-free queue. */
    std::uint32_t LogQueueCapacity() const { return logQueueCap_; }

    /** @brief Interval in milliseconds for the background thread to flush logs to disk. */
    std::uint32_t LogFlushIntervalMs() const { return logFlushMs_; }

    /** @brief Queue fill-level threshold that triggers an immediate background flush. */
    std::uint32_t LogNotifyThreshold() const { return logNotify_; }

    // [Metrics]

    /** @brief Moving average window size (in frames) for telemetry metrics. */
    std::uint32_t MetricsWindow() const { return metricsWindow_; }

    /** @brief Interval in milliseconds at which metrics are dumped to the log/console. */
    std::uint32_t MetricsPrintEveryMs() const { return metricsPrintMs_; }

    /**
     * @brief Calculates the absolute minimum number of slots required to prevent deadlocks.
     *
     * @details A deadlock occurs if all threads and queues are holding a slot, and a producer
     * tries to acquire a new one from an empty pool.
     * Formula: `PrepThreads + QPrepCapacity + InferThreads + QInfCapacity + PostThreads`
     *
     * @return The theoretical minimum slot capacity.
     */
    std::uint32_t MinimumSlots() const;

private:
    /**
     * @brief Private constructor to enforce creation only via the static factory method.
     */
    AppConfig(PipelineConfig model, PatchGeometry geometry);

    /**
     * @brief Validates that parameters do not contradict each other.
     * @throws std::runtime_error if `slots_ < MinimumSlots()` or if queue capacities are 0.
     */
    void ValidateCrossConstraints() const;

    PipelineConfig model_;
    PatchGeometry geometry_;

    std::uint32_t rawFrames_ = 0, slots_ = 0;
    std::uint32_t qRawCap_ = 0, qPrepCap_ = 0, qInfCap_ = 0;
    std::uint32_t prepThreads_ = 1, inferThreads_ = 1, postThreads_ = 1, openCvThreads_ = 1;
    std::uint32_t readTimeoutMs_ = 100;

    bool mapAtModelRes_ = true, drawMask_ = false;
    bool elevateProcess_ = true, prepRealtime_ = true, inferRealtime_ = true, postRealtime_ = false;

    std::uint32_t logMaxChars_ = 480, logQueueCap_ = 4096, logFlushMs_ = 200, logNotify_ = 1024;
    std::uint32_t metricsWindow_ = 100, metricsPrintMs_ = 2000;
};