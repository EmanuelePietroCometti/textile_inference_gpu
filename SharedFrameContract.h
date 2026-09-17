#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <cstdint>

/**
 * @file SharedFrameContract.h
 * @brief Mirror of the memory layout defined by the external producer.
 *
 * @details This file defines the strict data contract used to share memory
 * between the camera grabber process (producer) and the inference pipeline (consumer)
 * via Windows Memory Mapped Files (MMF). It must be strictly identical in both codebases.
 */

 /**
  * @brief Namespace encapsulating the Shared Frame Contract (SFC) definitions.
  */
namespace sfc {

    /// Number of frames expected in a single shared memory batch.
    inline constexpr UINT32 NUM_FRAMES = 16;

    /// Number of color channels per frame (e.g., 3 for RGB/BGR).
    inline constexpr UINT32 NUM_CHANNELS = 3;

    /// Byte size of a single frame (e.g., 512x512 resolution).
    inline constexpr UINT32 PAYLOAD = 512 * 512 * NUM_CHANNELS;

    /// Total byte size of the pixel payload within the shared memory block.
    inline constexpr UINT32 FULL_PAYLOAD = PAYLOAD * NUM_FRAMES;

    /// The unique system-wide name identifying the Memory Mapped File (MMF).
    inline constexpr wchar_t kMappingName[] = L"Local\\GrabBuffer_0";

    /// The unique system-wide name identifying the synchronization Event.
    inline constexpr wchar_t kReadyEventName[] = L"Local\\GrabBuffer_0_Ready";

    /**
     * @brief The exact memory layout mapped into the Shared Memory space.
     *
     * @details This structure relies on a lock-free atomic state machine to prevent
     * frame tearing and race conditions. The initial state is established by zeroing
     * the memory section during allocation by the OS, not by a C++ constructor.
     *
     * **Synchronization Protocol:**
     * - **Producer:**
     *   Executes `InterlockedCompareExchange(&canRead, TRUE, FALSE)`.
     *   If it returns `FALSE` (meaning the buffer was free), the producer copies the
     *   pixel data into `payload_` and then fires `SetEvent(ready)`.
     * - **Consumer (This Pipeline):**
     *   Waits via `WaitForSingleObject(ready)`. Once awoken, it reads/copies the data
     *   from `payload_` and immediately calls `InterlockedExchange(&canRead, FALSE)`
     *   to release the buffer back to the producer.
     */
    struct Frame {
        /**
         * @brief Atomic flag indicating ownership of the buffer.
         *
         * @details `FALSE` (0) = Free, producer can write. `TRUE` (1) = Ready, consumer can read.
         * Marked `volatile` to prevent compiler optimization across threads and processes,
         * ensuring strict memory visibility in conjunction with Interlocked functions.
         */
        volatile LONG canRead;

        /**
         * @brief Contiguous raw memory block holding the pixel data for the entire batch.
         */
        uint8_t payload_[FULL_PAYLOAD];
    };

} // namespace sfc