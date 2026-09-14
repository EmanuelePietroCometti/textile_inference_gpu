#pragma once

#include <vector>
#include <mutex>
#include <cstddef>

/**
 * @brief Thread-safe circular buffer for computing a Simple Moving Average (SMA).
 *
 * @details This class maintains a fixed-size window of recent double-precision floating-point
 * values. Once the window reaches its maximum capacity, new values overwrite the oldest ones
 * in a circular fashion. It is completely thread-safe, making it ideal for tracking real-time
 * metrics across multiple threads (e.g., pipeline FPS, inference latency, or CPU load).
 */
class RollingField
{
public:
    /**
     * @brief Constructs the rolling field with a specific window size.
     *
     * @param windowSize The maximum number of samples to keep in the moving window.
     */
    explicit RollingField(std::size_t windowSize);

    /**
     * @brief Inserts a new value into the moving window.
     *
     * @details If the window is already full, this operation will overwrite the oldest
     * sample currently stored, maintaining the circular nature of the buffer.
     *
     * @param value The new data point to add to the rolling window.
     */
    void add(double value);

    /**
     * @brief Calculates the average of the currently stored samples.
     *
     * @details If no samples have been added yet, this method returns 0.0.
     * It only computes the average using the actual number of inserted elements
     * (up to `windowSize`), ignoring empty slots during the initial ramp-up phase.
     *
     * @return The arithmetic mean of the recorded samples.
     */
    double average() const;

    /**
     * @brief Retrieves the current number of active samples contributing to the average.
     *
     * @return The number of elements (ranging from 0 to `windowSize`).
     */
    std::size_t samples() const;

    /**
     * @brief Clears the window, logically erasing all stored samples.
     *
     * @details Resets the internal write index and count to zero. It does not deallocate
     * the underlying vector memory, keeping future insertions allocation-free.
     */
    void clear();

private:
    std::vector<double> window;   ///< Pre-allocated circular buffer holding the samples.
    mutable std::mutex mutex;     ///< Mutex protecting internal state. Mutable to allow locking in const methods.
    std::size_t writeIndex = 0;   ///< The index where the next incoming sample will be written.
    std::size_t count = 0;        ///< The current number of valid samples stored (capped at windowSize).
};