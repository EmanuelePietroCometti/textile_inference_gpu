#pragma once

#include <atomic>
#include <cstdint>
#include "RollingAverage.h" // Ensures access to the thread-safe RollingField class

/**
 * @brief Aggregates and tracks real-time latency metrics for a batched GPU inference pipeline.
 *
 * @details This class provides a centralized, thread-safe telemetry hub. It uses multiple
 * `RollingField` instances to independently track the moving average of time spent in
 * various stages of the pipeline (Preprocessing, Memory Transfers, Execution, and Postprocessing).
 * This fine-grained tracking is crucial for identifying bottlenecks (e.g., PCIe bandwidth limits
 * vs. Compute limits) in high-performance computer vision applications.
 */
class PerformanceMetrics {
public:
    /**
    * @brief Non-atomic snapshot of all current moving averages and the batch counter.
    *
    * @details This structure holds a point-in-time copy of the metrics. Since the individual 
    * `RollingField` averages are fetched sequentially without a global lock, the snapshot 
    * is not strictly atomic across all fields. This is an intentional design choice to prevent 
    * telemetry logging from blocking the critical real-time execution threads.
    */
    struct Snapshot {
        double preprocessing = 0.0;     ///< Moving average of CPU image preparation latency.
        double gpu = 0.0;               ///< Moving average of total GPU occupancy latency.
        double h2d = 0.0;               ///< Moving average of Host-to-Device PCIe transfer latency.
        double run = 0.0;               ///< Moving average of CUDA/TensorRT kernel execution latency.
        double d2h = 0.0;               ///< Moving average of Device-to-Host PCIe transfer latency.
        double postprocessing = 0.0;    ///< Moving average of CPU bounding box/mask extraction latency.
        int64_t completedBatches = 0;   ///< Total number of successfully processed batches at the time of the snapshot.
    };

    /**
     * @brief External figures needed by the final report, owned by other modules.
     */
    struct FinalReportInput {
        std::uint64_t framesRead = 0;        ///< Frames accepted from the MMF.
        std::uint64_t framesDropped = 0;     ///< Frames discarded because the raw pool was empty.
        std::uint64_t lostLogMessages = 0;   ///< Log records dropped by the async logger.
        double listenedSec = 0.0;            ///< Total time spent listening (sum of every s ... x interval).
        std::uint32_t prepThreads = 1;       ///< Threads running the prep stage.
        std::uint32_t inferThreads = 1;      ///< Threads running the infer stage.
        std::uint32_t postThreads = 1;       ///< Threads running the post stage.
    };
    
    /**
     * @brief Constructs the metrics tracker.
     *
     * @param windowDimension Number of recent samples kept by every moving average. It is also
     * the period of `printRollingAverage`: one `[MONITOR]` line every `windowDimension` batches.
     */
    explicit PerformanceMetrics(int windowDimension);

    /**
     * @brief Destructor. Safely cleans up the metrics tracker.
     */
    ~PerformanceMetrics();

    /**
     * @brief Logs the time taken to decode, resize, or normalize input data on the CPU.
     * @param t The elapsed time (typically in milliseconds).
     */
    void addPreprocessingTime(double t);

    /**
     * @brief Logs the total cumulative time the GPU was active for a batch.
     * @param t The elapsed time.
     */
    void addGpuTime(double t);

    /**
     * @brief Logs the Host-to-Device (H2D) memory transfer time (e.g., CPU RAM to VRAM over PCIe).
     * @param t The elapsed time.
     */
    void addH2DTime(double t);

    /**
     * @brief Logs the actual neural network inference execution time on the GPU (Compute).
     * @param t The elapsed time.
     */
    void addRunTime(double t);

    /**
     * @brief Logs the Device-to-Host (D2H) memory transfer time (e.g., VRAM to CPU RAM over PCIe).
     * @param t The elapsed time.
     */
    void addD2HTime(double t);

    /**
     * @brief Logs the post-processing time and counts the batch as completed.
     *
     * @details This is the last metric recorded for a batch, so it also advances the completed
     * batch counter. The increment is a single atomic read-modify-write: every call, from any
     * thread, gets a distinct index.
     *
     * @param t The elapsed time.
     * @return The 1-based index of the batch just completed, unique across all calling threads.
     */
    std::int64_t addPostprocessingTime(double t);

    /**
     * @brief Logs the current moving averages, once every `windowDimension` completed batches.
     *
     * @details Meant to be called right after `addPostprocessingTime`, passing its return value.
     * Because that index is unique per call, with N post-processing threads exactly one thread
     * sees each multiple of the window and prints: no duplicate lines, none skipped. The
     * function never re-reads the shared counter, which is what made the previous version racy.
     * Cost on the other calls: one modulo.
     *
     * @param batchIndex The value returned by `addPostprocessingTime` for the batch just completed.
     */
     void printRollingAverage(std::int64_t batchIndex) const;

    /**
     * @brief Retrieves a point-in-time snapshot of the current performance metrics.
     *
     * @details Extracts the current arithmetic mean from all underlying `RollingField` instances
     * and copies the atomic batch counter, packing them into a single, easy-to-read structure.
     *
     * @return A `Snapshot` object containing the latest latency averages and batch count.
     */
    Snapshot snapshot() const;

    /**
     * @brief Logs the end-of-run summary, one Log::Info per line.
     *
     * @details One record per line keeps each line timestamped and far below the logger's
     * maxMessageChars. Call it after the stages are drained: nothing else is logging by then,
     * so the lines come out contiguous. Means cover the last min(windowDimension, completed) batches.
     */
    void logFinalReport(const FinalReportInput& in) const;
private:
    int windowDimension = 0;                     ///< Capacity of the rolling windows and print period, in batches.
    std::atomic<int64_t> completedBatches;       ///< Lock-free counter tracking total successfully processed batches.

    // Independent thread-safe moving average trackers for each pipeline stage
    RollingField preprocessing;    ///< Tracks CPU image preparation latency.
    RollingField gpu;              ///< Tracks total GPU occupancy latency.
    RollingField h2d;              ///< Tracks Host-to-Device PCIe transfer latency.
    RollingField run;              ///< Tracks CUDA/TensorRT kernel execution latency.
    RollingField d2h;              ///< Tracks Device-to-Host PCIe transfer latency.
    RollingField postprocessing;   ///< Tracks CPU bounding box/mask extraction latency.
};