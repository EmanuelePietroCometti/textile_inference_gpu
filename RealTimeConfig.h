#pragma once

#include <vector>
#include <Windows.h>

/**
 * @brief RAII-style scope guard for temporarily enabling Windows real-time execution priorities.
 *
 * @details This class is designed to elevate the calling process/thread to real-time or
 * high priority and increase the Windows multimedia timer resolution (e.g., via `timeBeginPeriod`).
 * It ensures strict CPU scheduling for latency-critical paths (like GPU inference, memory-mapped
 * file reading, or frame capturing) to minimize jitter and prevent OS context-switch interruptions.
 * The original scheduling state is automatically restored upon destruction.
 */
class RealTimeScope {
public:
    /**
     * @brief Constructs the scope, attempting to elevate execution priority and system timer resolution.
     */
    RealTimeScope();

    /**
     * @brief Destructor. Safely restores the previous priority class and timer resolution.
     */
    ~RealTimeScope();

    // Disable copy semantics to guarantee strict RAII resource management.
    RealTimeScope(const RealTimeScope&) = delete;
    RealTimeScope& operator=(const RealTimeScope&) = delete;

    /**
     * @brief Checks if the process/thread priority was successfully elevated.
     *
     * @return `true` if priority was elevated (e.g., HIGH_PRIORITY_CLASS), `false` otherwise.
     */
    bool elevated() const noexcept;

    /**
     * @brief Checks if strict real-time priority was successfully achieved.
     *
     * @return `true` if operating in strict real-time mode (e.g., REALTIME_PRIORITY_CLASS), `false` otherwise.
     */
    bool realtime() const noexcept;

private:
    DWORD previousClass = 0;   ///< Stores the original Windows priority class to restore on destruction.
    bool timerRaised = false;  ///< Flag indicating if the system multimedia timer resolution was successfully raised (e.g., to 1ms).
    bool realtime_ = false;    ///< Flag indicating if REALTIME_PRIORITY_CLASS was successfully acquired.
    bool elevated_ = false;    ///< Flag indicating if at least a high-priority class was acquired.
};

/**
 * @brief Utility namespace for thread and process CPU scheduling configurations.
 */
namespace RT {
    /**
     * @brief Configures the calling thread for background execution.
     *
     * @details Lowers the thread's scheduling priority (e.g., to `THREAD_PRIORITY_LOWEST` or
     * `THREAD_MODE_BACKGROUND_BEGIN`). Ideal for offloading non-critical tasks like asynchronous
     * logging to prevent them from stealing CPU cycles from the critical path.
     *
     * @return `0` on success, otherwise the code from `GetLastError()`.
     */
    [[nodiscard]] DWORD ConfigureBackgroundThread();

    /**
     * @brief Configures the calling thread for strict real-time execution.
     *
     * @details Elevates the thread's scheduling priority (e.g., to `THREAD_PRIORITY_TIME_CRITICAL`).
     * Use this exclusively for the hottest paths in the pipeline to guarantee microsecond-level
     * scheduling accuracy.
     *
     * @return `0` on success, otherwise the code from `GetLastError()`.
     */
    [[nodiscard]] DWORD ConfigureRealtimeThread();

    /**
     * @brief Sets the calling thread to an explicit Windows priority level.
     *
     * @details The effective base priority is class + level: under REALTIME_PRIORITY_CLASS
     * 15 -> 31, 2 -> 26, 1 -> 25, 0 -> 24, -1 -> 23, -2 -> 22, -15 -> 16; under
     * HIGH_PRIORITY_CLASS 15 -> 15, 2 -> 15, 1 -> 14, 0 -> 13, -1 -> 12, -2 -> 11, -15 -> 1.
     *
     * @param priority One of the THREAD_PRIORITY_* levels (see IsValidThreadPriority).
     * @return `0` on success, otherwise the code from `GetLastError()`.
     */
    [[nodiscard]] DWORD ConfigureThread(int priority);

    /**
     * @brief True for the THREAD_PRIORITY_* levels valid in every priority class:
     * -15 (IDLE), -2, -1, 0, 1, 2, 15 (TIME_CRITICAL).
     */
    [[nodiscard]] bool IsValidThreadPriority(int priority) noexcept;

    /**
     * @brief Applies a specific priority class to the current thread or process.
     *
     * @details Allows fine-grained, manual control over the Windows scheduler priority classes
     * (e.g., `HIGH_PRIORITY_CLASS`, `ABOVE_NORMAL_PRIORITY_CLASS`).
     *
     * @param requested The Windows API priority class flag to apply.
     * @return A `DWORD` representing the previous priority class before the change was applied.
     */
    DWORD ApplyPriorityClass(DWORD requested);
}