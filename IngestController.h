#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <thread>

class MmfFrameSource;

/**
 * @brief Owns the ingest thread: one thread per listening session (s ... x).
 *
 * @details Start() re-arms the MMF source and spawns the thread; Stop() signals the
 * source and joins it. The stage threads are NOT touched: while not listening they
 * simply block on an empty qRaw at zero CPU cost.
 */
class IngestController {
public:
    IngestController(MmfFrameSource& source, std::uint32_t readTimeoutMs, int priority);

    ~IngestController();

    IngestController(const IngestController&) = delete;
    IngestController& operator=(const IngestController&) = delete;

    bool Listening() const { return thread_.joinable(); }

    /// False if the thread exited on its own (e.g. WaitForSingleObject failure) while still "listening".
    bool Alive() const { return alive_.load(std::memory_order_acquire); }

    void Start();

    void Stop();

    double ListenedSec();
private:
    MmfFrameSource& source_;
    const std::uint32_t readTimeoutMs_;
    const int priority_;
    std::thread thread_;
    std::atomic<bool> alive_{ false };

    std::chrono::high_resolution_clock::time_point listenStart_{};      ///< Start of the session in progress (valid while Listening()).
    double listenedSec_ = 0.0;                                          ///< Sum of the sessions already closed by Stop().

};
