#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "OrtSession.h"      // OrtSessionConfig: geometry read from the model
#include "RingBuffer.h"

/**
 * @brief Data container for a single in-flight inference batch. No internal logic, no locks.
 *
 * @details
 * **Ownership Model:**
 * Ownership is strictly implicit: whichever thread holds the pointer to the slot
 * inherently owns it. Transitions between pipeline stages are managed exclusively by
 * pushing/popping from thread-safe `RingBuffer` queues. Duplicating this state with an
 * internal mutex or a per-slot state machine would create two competing sources of truth,
 * leading to divergence and race conditions in a multi-threaded architecture.
 *
 * **Memory Architecture:**
 * `input`, `scores`, and `rawMap` are allocated as CUDA **Pinned** (page-locked) memory,
 * but they are intentionally **not mapped** into the GPU's address space. While mapped memory
 * would theoretically eliminate the need for `cudaMemcpyAsync`, it would force the GPU to
 * fetch massive payloads (e.g., 51 MiB) on-demand over the PCIe bus during kernel execution,
 * resulting in highly fragmented memory accesses and severe latency. Using standard pinned memory
 * allows for a full-bandwidth, overlapping DMA burst transfer prior to kernel execution.
 *
 * `outMap` and `outMask` are never transferred to the GPU (no H2D), so they safely reside
 * in standard pageable host memory.
 */
struct PipelineSlot {
    float* input = nullptr;          ///< Pinned [B*C*H*W], populated by the preprocessing stage.
    float* scores = nullptr;         ///< Pinned [B], target buffer for the D2H transfer of classification scores.
    float* rawMap = nullptr;         ///< Pinned [B*mH*mW], target buffer for the D2H transfer of raw segmentation logits.
    std::uint8_t* outMap = nullptr;  ///< Standard host memory, normalized 8-bit map produced by post-processing.
    std::uint8_t* outMask = nullptr; ///< Standard host memory, binary mask. Remains `nullptr` if disabled.

    std::uint64_t seq = 0;           ///< Sequence number tracking the frame across the pipeline.
    std::int64_t acquiredQPC = 0;    ///< High-resolution acquisition timestamp propagated from the `RawFrame`.

    // Telemetry Timings
    double prepMs = 0.0;             ///< CPU Preprocessing latency.
    double h2dMs = 0.0;              ///< Host-to-Device transfer latency.
    double runMs = 0.0;              ///< GPU Kernel execution latency.
    double d2hMs = 0.0;              ///< Device-to-Host transfer latency.
    double postMs = 0.0;             ///< CPU Post-processing latency.
};

/**
 * @brief Pre-allocated memory pool managing the lifecycle of `PipelineSlot` instances.
 *
 * @details This pool acts as the memory foundation for the entire inference pipeline,
 * guaranteeing a strict **zero-allocation hot path**. All pinned and pageable memory is
 * allocated during construction.
 *
 * **Lifecycle & Thread Safety:**
 * The pool exposes available slots via a `RingBuffer`.
 * 1. Preprocessing `pop()`s an empty slot, fills the `input`, and pushes it to the inference queue.
 * 2. Inference executes and pushes it to the post-processing queue.
 * 3. Post-processing finalizes the data and `push()`es the slot back into this pool.
 *
 * *Crucial Rule:* Post-processing is the sole release point. A slot must be returned to the pool
 * even if an exception occurs during earlier stages. Always use an RAII scope guard to return slots,
 * rather than relying on an instruction at the end of a `try` block.
 *
 * **Capacity Sizing:**
 * To prevent pipeline stalls, `SlotCount` must be strictly greater than or equal to:
 * `PrepThreads + cap(qPrep) + InferenceThreads + cap(qInf) + PostThreads`
 */
class SlotStore {
public:
    /**
     * @brief Constructs the memory store and pre-allocates all necessary slot buffers.
     *
     * @param session A fully initialized `OrtSessionConfig`. The required tensor geometries
     *                (batch, channels, height, width) are extracted directly from the model,
     *                preventing configuration mismatches.
     * @param slotCount Total number of `PipelineSlot` objects to allocate.
     * @param outMapElemsPerImage Total elements for the output map per single image
     *                            (e.g., `modelH * modelW` or `patchH * patchW`).
     * @param wantMask If `false`, `outMask` pointers remain `nullptr` and their memory is not allocated.
     * @throws std::runtime_error if a `cudaHostAlloc` fails (e.g., out of memory). Any allocations
     *         made prior to the failure are safely rolled back.
     */
    SlotStore(const OrtSessionConfig& session,
        std::size_t slotCount,
        std::size_t outMapElemsPerImage,
        bool wantMask);

    ~SlotStore();

    // Disable copy semantics to ensure unique ownership of OS/CUDA memory resources.
    SlotStore(const SlotStore&) = delete;
    SlotStore& operator=(const SlotStore&) = delete;

    /**
     * @brief Accesses the thread-safe pool of available pipeline slots.
     * @return A reference to the underlying `RingBuffer` managing `PipelineSlot*`.
     */
    RingBuffer<PipelineSlot*>& Pool() { return pool_; }

    /** @brief Retrieves the total capacity of the pool. */
    std::size_t SlotCount() const { return slots_.size(); }

    /** @brief Retrieves the amount of CUDA pinned memory utilized by a single slot. */
    std::size_t PinnedBytesPerSlot() const { return pinnedPerSlot_; }

    /** @brief Retrieves the total amount of CUDA pinned memory allocated by the store. */
    std::size_t TotalPinnedBytes() const { return pinnedPerSlot_ * slots_.size(); }

    /** @brief Retrieves the configured number of elements for a single image's output map. */
    std::size_t OutMapElemsPerImage() const { return outMapElems_; }

private:
    /** @brief Safely releases all CUDA pinned memory and host memory. */
    void Cleanup();

    /**
     * @brief Helper to allocate page-locked memory via CUDA.
     * @param elems Number of `float` elements to allocate.
     * @return A pointer to the pinned memory block.
     * @throws std::runtime_error if allocation fails.
     */
    float* AllocPinnedFloat(std::size_t elems);

    std::vector<PipelineSlot> slots_;                             ///< Stable storage for the slot metadata.
    std::vector<void*> pinnedBlocks_;                             ///< Tracks raw pointers for safe `cudaFreeHost` execution.
    std::vector<std::unique_ptr<std::uint8_t[]>> hostBlocks_;     ///< RAII containers for pageable host memory (outMap / outMask).
    RingBuffer<PipelineSlot*> pool_;                              ///< Thread-safe circular queue holding pointers to free slots.

    std::size_t pinnedPerSlot_ = 0;                               ///< Cached size of pinned memory per slot.
    std::size_t outMapElems_ = 0;                                 ///< Cached element count for output maps.
};