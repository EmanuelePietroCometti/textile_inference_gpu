#include "StageWorkers.h"

StageWorkers::StageWorkers(RingBuffer<RawFrame*>& qRaw, RingBuffer<RawFrame*>& rawPool,
    RingBuffer<PipelineSlot*>& qPrep, RingBuffer<PipelineSlot*>& qInf,
    RingBuffer<PipelineSlot*>& slotPool)
    : qRaw_(qRaw), rawPool_(rawPool), qPrep_(qPrep), qInf_(qInf), slotPool_(slotPool) {
}

StageWorkers::~StageWorkers() { Abort(); }

void StageWorkers::Drain()
{
    qRaw_.stop();  JoinAll(prep);
    qPrep_.stop(); JoinAll(infer);
    qInf_.stop();  JoinAll(post);
    // Pools are stopped last: during the drain prep may still be waiting for a
    // slot that a post thread is about to give back.
    slotPool_.stop();
    rawPool_.stop();
}

void StageWorkers::Abort()
{
    qRaw_.stop(); qPrep_.stop(); qInf_.stop();
    slotPool_.stop(); rawPool_.stop();
    JoinAll(prep); JoinAll(infer); JoinAll(post);
}

void StageWorkers::JoinAll(std::vector<std::thread>& ts)
{
    for (auto& t : ts) if (t.joinable()) t.join();
}
