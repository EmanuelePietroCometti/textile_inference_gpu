#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "PatchLayout.h"
#include "RingBuffer.h"

/**
 * @brief A lightweight container representing a single captured image strip and its sub-patches.
 *
 * @details This structure holds the raw pixel data and pre-configured OpenCV matrix headers
 * (`cv::Mat`) that point directly into the `strip` memory. This enables zero-copy patch
 * extraction. It also carries essential metadata like the sequence number and hardware timestamp.
 */
struct RawFrame {
    std::uint8_t* strip = nullptr;   ///< Raw pointer to the contiguous pixel buffer (managed by RawFrameStore).
    std::vector<cv::Mat> patches;    ///< Collection of zero-copy OpenCV headers pointing to specific regions of `strip`.
    std::uint64_t seq = 0;           ///< Monotonically increasing sequence number of the captured frame.
    std::int64_t acquiredQPC = 0;    ///< High-resolution hardware timestamp (QueryPerformanceCounter) at acquisition time.
};

/**
 * @brief Pre-allocated memory pool managing a fixed number of `RawFrame` instances.
 *
 * @details This class is the core memory manager for the image acquisition pipeline.
 * It pre-allocates all required heap memory for the image strips and `cv::Mat` headers
 * during initialization. By distributing `RawFrame*` pointers via a thread-safe `RingBuffer`,
 * it guarantees a completely allocation-free (zero-alloc) hot path during real-time camera capture.
 */
class RawFrameStore {
public:
    /**
     * @brief Constructs the memory store and pre-allocates all buffers.
     *
     * @param layout The geometry configuration detailing strip dimensions and patch offsets.
     * @param frameCount The maximum number of frames to keep in memory simultaneously (pool capacity).
     */
    RawFrameStore(const PatchLayout& layout, std::size_t frameCount);

    /**
     * @brief Destructor. Safely cleans up the pre-allocated memory blocks.
     */
    ~RawFrameStore();

    // Disable copy semantics to ensure unique ownership of the underlying memory buffers.
    RawFrameStore(const RawFrameStore&) = delete;
    RawFrameStore& operator=(const RawFrameStore&) = delete;

    /**
     * @brief Accesses the thread-safe pool of available frame pointers.
     *
     * @details Producer threads (e.g., Camera grabber) pop pointers from this pool, populate them,
     * and pass them downstream. Consumer threads must push the pointers back into this pool
     * once processing is complete to recycle the memory.
     *
     * @return A reference to the underlying `RingBuffer` managing `RawFrame*`.
     */
    RingBuffer<RawFrame*>& Pool();

    /**
     * @brief Retrieves the total capacity of the store.
     * @return The number of frames managed by this store.
     */
    std::size_t FrameCount() const;

    /**
     * @brief Retrieves the memory size required for a single full strip.
     * @return The number of bytes per frame.
     */
    std::size_t BytesPerFrame() const;

    /**
     * @brief Retrieves the total heap memory pre-allocated by the store for all strips.
     * @return The total size in bytes.
     */
    std::size_t TotalBytes() const;

    /**
     * @brief Retrieves the geometry layout used to configure these frames.
     * @return A constant reference to the underlying `PatchLayout`.
     */
    const PatchLayout& Layout() const;

private:
    PatchLayout layout_;                                     ///< Copy of the layout configuration used for pointer arithmetic.
    std::vector<std::unique_ptr<std::uint8_t[]>> storage_;   ///< RAII containers holding the actual huge contiguous heap allocations.
    std::vector<RawFrame> frames_;                           ///< Stable storage for the frame objects themselves.
    RingBuffer<RawFrame*> pool_;                             ///< Thread-safe circular queue holding pointers to available, recyclable frames.
};