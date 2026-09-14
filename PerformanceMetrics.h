#pragma once

#include <atomic>
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
        double batchPrep = 0.0;         ///< Moving average of CPU batch memory alignment latency.
        double gpu = 0.0;               ///< Moving average of total GPU occupancy latency.
        double h2d = 0.0;               ///< Moving average of Host-to-Device PCIe transfer latency.
        double run = 0.0;               ///< Moving average of CUDA/TensorRT kernel execution latency.
        double d2h = 0.0;               ///< Moving average of Device-to-Host PCIe transfer latency.
        double postprocessing = 0.0;    ///< Moving average of CPU bounding box/mask extraction latency.
        int64_t completedBatches = 0;   ///< Total number of successfully processed batches at the time of the snapshot.
    };
    /**
     * @brief Constructs the metrics tracker with specific rolling window limits and batch dimensions.
     *
     * @param windowDimension The maximum number of recent samples to keep for each metric's moving average.
     * @param batchSize The number of frames/images processed in a single inference batch.
     */
    PerformanceMetrics(int windowDimension, int batchSize);

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
     * @brief Logs the time taken to assemble individual preprocessed frames into a contiguous memory batch.
     * @param t The elapsed time.
     */
    void addBatchPrepTime(double t);

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
     * @brief Logs the time taken to parse network outputs (e.g., NMS, bounding box scaling) on the CPU.
     * @param t The elapsed time.
     */
    void addPostprocessingTime(double t);

    /**
     * @brief Outputs the current moving averages to the standard output or logger.
     *
     * @param window The specific number of recent completed batches to consider for the printout
     * (can be independent of the `windowDimension` capacity).
     */
    void printRollingAverage(int window);

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
     * @brief Resets all tracked metrics and the completed batch counter back to zero.
     */
    void clear();

private:
    int windowDimension = 0;                     ///< The capacity of the rolling windows.
    int batchSize = 0;                           ///< The batch size used by the underlying deep learning model.
    std::atomic<int64_t> completedBatches;       ///< Lock-free counter tracking total successfully processed batches.

    // Independent thread-safe moving average trackers for each pipeline stage
    RollingField preprocessing;    ///< Tracks CPU image preparation latency.
    RollingField batchPrep;        ///< Tracks CPU batch memory alignment latency.
    RollingField gpu;              ///< Tracks total GPU occupancy latency.
    RollingField h2d;              ///< Tracks Host-to-Device PCIe transfer latency.
    RollingField run;              ///< Tracks CUDA/TensorRT kernel execution latency.
    RollingField d2h;              ///< Tracks Device-to-Host PCIe transfer latency.
    RollingField postprocessing;   ///< Tracks CPU bounding box/mask extraction latency.
};