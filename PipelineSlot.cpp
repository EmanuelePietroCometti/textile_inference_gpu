#include "PipelineConfig.h"


#include "PipelineSlot.h"
#include "AsyncLogger.h"

#include <cstring>
#include <stdexcept>

namespace {

    /**
     * @brief Validates the requested slot capacity before RingBuffer construction.
     * 
     * @details A pool cannot have zero capacity. This validation must occur *before* 
     * the initializer list executes to prevent the `RingBuffer` from throwing or 
     * allocating invalid internal states.
     * 
     * @param n The requested number of slots.
     * @return The validated number of slots.
     * @throws std::invalid_argument if the requested capacity is zero.
     */
    std::size_t CheckedSlotCount(std::size_t n)
    {
        if (n == 0)
            throw std::invalid_argument("SlotStore requires at least one slot capacity.");
        return n;
    }

} // namespace

float* SlotStore::AllocPinnedFloat(std::size_t elems)
{
    void* p = nullptr;
    
    // We explicitly use cudaHostAllocDefault instead of cudaHostAllocMapped.
    // We want the DMA controller to execute a full-bandwidth burst transfer via 
    // cudaMemcpyAsync, rather than forcing the GPU kernel to fetch memory on-demand 
    // over the PCIe bus, which causes severe latency.
    const cudaError_t err = cudaHostAlloc(&p, elems * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaHostAlloc of "
            + std::to_string(elems * sizeof(float)) + " bytes failed: "
            + cudaGetErrorString(err));
    }

    pinnedBlocks_.push_back(p);
    
    // Page Fault Warming: 
    // Touch every page immediately. First-access page faults must be resolved now 
    // rather than during the first real inference batch, which would heavily skew 
    // the latency metrics.
    std::memset(p, 0, elems * sizeof(float));
    
    return static_cast<float*>(p);
}

SlotStore::SlotStore(const OrtSessionConfig& session,
                     std::size_t slotCount,
                     std::size_t outMapElemsPerImage,
                     bool wantMask)
    : pool_(CheckedSlotCount(slotCount)),
      outMapElems_(outMapElemsPerImage)
{
    const std::size_t inputElems = session.InputElems();
    const std::size_t scoreElems = session.ScoreElems();
    const std::size_t rawMapElems = session.MapElems();
    const std::size_t outMapBytes =
        static_cast<std::size_t>(session.BatchSize()) * outMapElemsPerImage;

    pinnedPerSlot_ = (inputElems + scoreElems + rawMapElems) * sizeof(float);

    try {
        pinnedBlocks_.reserve(slotCount * 3);
        hostBlocks_.reserve(slotCount * 2);
        
        // CRITICAL: Resize the slots vector BEFORE taking addresses. 
        // A subsequent push_back would reallocate the vector's internal array, 
        // silently invalidating all pointers we've already pushed into the pool.
        slots_.resize(slotCount);

        for (std::size_t k = 0; k < slotCount; ++k) {
            PipelineSlot& s = slots_[k];

            s.input  = AllocPinnedFloat(inputElems);
            s.scores = AllocPinnedFloat(scoreElems);
            s.rawMap = AllocPinnedFloat(rawMapElems);

            // Using C++20 make_unique_for_overwrite avoids the overhead of default-initializing
            // the array twice (once by unique_ptr, once by memset).
            hostBlocks_.push_back(std::make_unique_for_overwrite<std::uint8_t[]>(outMapBytes));
            s.outMap = hostBlocks_.back().get();
            std::memset(s.outMap, 0, outMapBytes);

            if (wantMask) {
                hostBlocks_.push_back(std::make_unique_for_overwrite<std::uint8_t[]>(outMapBytes));
                s.outMask = hostBlocks_.back().get();
                std::memset(s.outMask, 0, outMapBytes);
            }

            if (!pool_.try_push(&s)) {
                throw std::runtime_error("SlotStore: Internal pool capacity is smaller than slotCount.");
            }
        }
    }
    catch (...) {
        // Exception Safety (RAII Rollback):
        // If an exception occurs during construction (e.g., Out Of Memory), the destructor 
        // is NOT called. We must manually clean up the successfully executed cudaHostAlloc 
        // calls to prevent a permanent GPU pinned memory leak.
        Cleanup();
        throw;
    }

    Log::Info("SlotStore ready | {} slots | pinned {:.1f} MiB/slot, {:.1f} MiB total | outMap {} B/img | mask={}",
        slotCount,
        pinnedPerSlot_ / (1024.0 * 1024.0),
        TotalPinnedBytes() / (1024.0 * 1024.0),
        outMapElemsPerImage, wantMask);
}

SlotStore::~SlotStore()
{
    pool_.stop();   // Unblocks any thread currently waiting on a pool_.pop()
    Cleanup();
}

void SlotStore::Cleanup()
{
    // Defensive Programming: 
    // Nullify the slot pointers before freeing the underlying memory. If a downstream 
    // thread holds a dangling pointer and attempts a use-after-free, it will instantly 
    // trigger a clean Segmentation Fault rather than silently reading corrupted, recycled memory.
    for (PipelineSlot& s : slots_) {
        s.input = nullptr; s.scores = nullptr; s.rawMap = nullptr;
        s.outMap = nullptr; s.outMask = nullptr;
    }
    
    // Release hardware resources
    for (void* p : pinnedBlocks_) {
        if (p) cudaFreeHost(p);
    }
    
    pinnedBlocks_.clear();
    hostBlocks_.clear();
}