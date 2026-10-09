#pragma once

#include <thread>
#include <vector>

#include "PipelineSlot.h"
#include "RawFrame.h"
#include "RingBuffer.h"

/**
 * @brief Owns the prep/infer/post threads and guarantees they are joined.
 *
 * @details Drain(): graceful stop, queues are stopped one stage at a time, in pipeline
 * order. RingBuffer::pop keeps returning items until the queue is both stopped AND
 * empty, so every frame already in qRaw reaches the sink.
 * Abort() (destructor, i.e. exception path): all queues stopped at once, then join.
 * No blocking wait survives a stopped queue, so the joins cannot hang even if a
 * stage is missing (e.g. thread creation failed halfway through).
 * Must be destroyed BEFORE the stores/queues it references: declare it after them.
 */
class StageWorkers {
public:
    StageWorkers(RingBuffer<RawFrame*>& qRaw, RingBuffer<RawFrame*>& rawPool,
        RingBuffer<PipelineSlot*>& qPrep, RingBuffer<PipelineSlot*>& qInf,
        RingBuffer<PipelineSlot*>& slotPool);

    ~StageWorkers();

    StageWorkers(const StageWorkers&) = delete;
    StageWorkers& operator=(const StageWorkers&) = delete;

    std::vector<std::thread> prep, infer, post;

    void Drain();

    void Abort();

private:
    static void JoinAll(std::vector<std::thread>& ts);

    RingBuffer<RawFrame*>& qRaw_;
    RingBuffer<RawFrame*>& rawPool_;
    RingBuffer<PipelineSlot*>& qPrep_;
    RingBuffer<PipelineSlot*>& qInf_;
    RingBuffer<PipelineSlot*>& slotPool_;
};
