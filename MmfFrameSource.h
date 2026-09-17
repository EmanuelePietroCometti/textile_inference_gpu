#pragma once
#include <Windows.h>
#include "SharedFrameContract.h"
#include "RingBuffer.h"
#include "RawFrame.h"

/**
 * @brief Represents the outcome of an attempt to read a frame from the shared memory.
 */
enum class FrameStatus {
    Ok,               ///< The frame was successfully read and pushed to the processing queue.
    TimeOut,          ///< The wait operation timed out before a new frame was signaled.
    Stopped,          ///< The source was explicitly stopped, interrupting the read attempt.
    GeometryMismatch  ///< The incoming frame dimensions in the MMF do not match the expected PatchLayout.
};

/**
 * @brief High-performance IPC (Inter-Process Communication) image consumer.
 *
 * @details This class connects to a Memory Mapped File (MMF) populated by an external
 * producer process (e.g., a camera grabber application). It waits for a Windows Event
 * to signal the arrival of new data, extracts the payload into a pre-allocated `RawFrame`
 * acquired from the object pool, and forwards it to the inference pipeline.
 */
class MmfFrameSource {
public:
    /**
     * @brief Constructs the MMF source and attaches to the underlying Windows IPC primitives.
     *
     * @param layout The expected geometry of the incoming frames. Used to validate
     *               incoming payloads and prevent buffer overflows.
     * @param rawPool Reference to the RingBuffer acting as the Object Pool for empty, recyclable frames.
     * @param outQueue Reference to the RingBuffer where successfully populated frames are pushed for inference.
     */
    MmfFrameSource(const PatchLayout& layout,
        RingBuffer<RawFrame*>& rawPool,
        RingBuffer<RawFrame*>& outQueue);

    /**
     * @brief Destructor. Safely closes the Windows handles (MMF and Event) and unmaps the memory view.
     */
    ~MmfFrameSource();

    // Disable copy semantics as this class manages unique OS-level handles and resources.
    MmfFrameSource(const MmfFrameSource&) = delete;
    MmfFrameSource& operator=(const MmfFrameSource&) = delete;

    /**
     * @brief Blocks and waits for a new frame to be written into the shared memory.
     *
     * @details This method waits on the Windows Event handle (`hEvent`). Once signaled,
     * it attempts to acquire a free `RawFrame` from `rawPool_`. If the pool is empty,
     * the frame is dropped (and the drop counter increments). If acquired, it copies the
     * payload and pushes it to `outQueue_`.
     *
     * @param timeoutMs The maximum time in milliseconds to wait for a new frame before returning `TimeOut`.
     * @return The status of the operation (`Ok`, `TimeOut`, `Stopped`, or `GeometryMismatch`).
     */
    FrameStatus ReadFrame(std::uint32_t timeoutMs);

    /**
     * @brief Signals the internal threads to halt operations.
     *
     * @details Sets the internal stop flag and forces any pending `WaitForSingleObject`
     * inside `ReadFrame` to unblock immediately, ensuring a clean and fast shutdown.
     */
    void Stop();

    /**
     * @brief Retrieves the total number of frames successfully read and passed to the pipeline.
     * @return The read frame count.
     */
    std::uint64_t FramesRead() const;

    /**
     * @brief Retrieves the number of frames dropped because the pipeline was too slow.
     *
     * @details A drop occurs when the MMF signals a new frame, but a call to `try_pop`
     * on the `rawPool_` fails because all pre-allocated frames are currently stuck
     * in the inference or post-processing stages.
     *
     * @return The dropped frame count.
     */
    std::uint64_t DroppedNoBuffer() const;

private:
    RingBuffer<RawFrame*>& rawPool_;   ///< Pool of empty buffers ready to receive data.
    RingBuffer<RawFrame*>& outQueue_;  ///< Pipeline queue for populated buffers ready for inference.

    HANDLE hMapping;  ///< Windows Handle to the File Mapping Object.
    void* pView;      ///< Pointer to the mapped memory address space.
    HANDLE hEvent;    ///< Windows Handle to the synchronization Event.

    sfc::Frame* frame_ = nullptr; ///< Casted pointer mapping the raw view to the agreed SharedFrameContract struct.

    std::uint64_t seq_ = 0;       ///< Internal tracker for the last processed sequence number.

    std::atomic<std::uint64_t> framesRead_{ 0 };      ///< Atomic counter for successful reads.
    std::atomic<std::uint64_t> droppedNoBuffer_{ 0 }; ///< Atomic counter for consumer-side drops.
    std::atomic<bool> stopped_{ false };              ///< Flag indicating if the source is shutting down.
};