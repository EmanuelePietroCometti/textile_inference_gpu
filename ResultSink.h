#pragma once

#include <cstddef>
#include <cstdint>

#include "AsyncLogger.h"

/**
 * @brief Data payload containing the finalized inference results for a single batch.
 *
 * @details
 * **CRITICAL MEMORY LIFETIME:**
 * The raw pointers (`scores`, `map`, `mask`) are strictly owned by the underlying
 * `PipelineSlot`. They are ONLY guaranteed to be valid for the synchronous duration
 * of the `IResultSink::Publish()` call. Immediately after `Publish()` returns, the
 * `PostStage` thread will recycle the slot into the pool, invalidating these pointers.
 * Any sink implementation that needs to defer processing or retain this data asynchronously
 * MUST perform a deep copy of the required memory blocks.
 */
struct BatchResult {
    std::uint64_t seq = 0;              ///< Monotonically increasing sequence number for chronological reordering.
    std::int64_t acquiredQPC = 0;       ///< High-resolution hardware timestamp captured at frame acquisition.
    int batch = 0;                      ///< The number of valid images processed in this batch.

    const float* scores = nullptr;      ///< Array of classification/anomaly scores [batch].
    const std::uint8_t* map = nullptr;  ///< Contiguous normalized segmentation maps [batch * mapElemsPerImage].
    const std::uint8_t* mask = nullptr; ///< Contiguous binary threshold masks [batch * mapElemsPerImage], or `nullptr` if disabled.
    std::size_t mapElemsPerImage = 0;   ///< Number of elements (pixels) per single image map.

    int anomalies = 0;                  ///< Total number of images in this batch that exceeded the score threshold.
};

/**
 * @brief Abstract interface defining the output destination for the pipeline's results.
 *
 * @details
 * **THREAD SAFETY & OUT-OF-ORDER EXECUTION:**
 * The `Publish` method will be called concurrently by multiple `PostStage` worker threads.
 * Because thread scheduling is non-deterministic, batches will likely arrive **out of order**
 * (e.g., sequence 5 might arrive before sequence 4). Implementations of this interface must
 * be internally thread-safe (e.g., using mutexes, concurrent queues, or lock-free structures)
 * and must utilize the `BatchResult::seq` field if strict chronological reordering is required
 * for downstream telemetry or UI rendering.
 */
class IResultSink {
public:
    virtual ~IResultSink() = default;

    /**
     * @brief Pushes the completed batch to the destination.
     * @param result The payload containing pointers to the processed data.
     */
    virtual void Publish(const BatchResult& result) = 0;
};

/**
 * @brief A no-op sink that immediately discards the payload.
 *
 * @details Extremely useful for pure hardware and throughput benchmarking.
 * Because the `PostStage` still performs all floating-point math, resizing, and memory
 * writes before invoking the sink, utilizing this sink guarantees that the `PerformanceMetrics`
 * accurately reflect real-world maximum load without being bottlenecked by disk I/O or UI rendering.
 */
class NullResultSink : public IResultSink {
public:
    void Publish(const BatchResult&) override {}
};

/**
 * @brief A basic diagnostic sink that outputs a text summary of the batch to the asynchronous logger.
 *
 * @details Ideal for initial pipeline bring-up, camera integration testing, or dry-run debugging
 * to verify that the MMF consumer, inference engine, and thresholds are working correctly
 * before attaching complex UI or database sinks.
 */
class LoggingResultSink : public IResultSink {
public:
    void Publish(const BatchResult& r) override
    {
        // Safe access: protects against out-of-bounds reads if a batch size of 0 is somehow passed.
        Log::Info("Batch {} | Anomalies {}/{} | score[0]={:.4f}",
            r.seq,
            r.anomalies,
            r.batch,
            r.batch > 0 ? r.scores[0] : 0.0f);
    }
};