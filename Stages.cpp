#include "Stages.h"
#include "AntialiasResizer.h"
#include "RealTimeConfig.h"

#include <chrono>
#include <cmath>
#include <vector>

namespace {

    double MsSince(const std::chrono::steady_clock::time_point& t0)
    {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }

    /**
     * @brief Releases the slot back to the pool regardless of how the scope exits.
     * @details This includes normal returns, loop breaks, and thrown exceptions.
     * It is the only safeguard against silent pool exhaustion.
     */
    class SlotGuard {
    public:
        SlotGuard(RingBuffer<PipelineSlot*>& pool, PipelineSlot* slot)
            : pool_(pool), slot_(slot) {
        }

        ~SlotGuard()
        {
            if (!slot_) return;
            if (!pool_.try_push(slot_)) {
                // Impossible by design: the slot comes from this pool, so there        
                // is enough space. If this happens, we are either shutting down        
                // (pool stopped) or there is a sizing error, and a lost slot cannot be recovered.        
                Log::Error("slot {} not returned to the pool: capacity error or shutting down", slot_->seq);
            }
            slot_ = nullptr;
        }

        SlotGuard(const SlotGuard&) = delete;
        SlotGuard& operator=(const SlotGuard&) = delete;

    private:
        RingBuffer<PipelineSlot*>& pool_;
        PipelineSlot* slot_;
    };

    /**
     * @brief Fused scale and offset per channel: out = u8 * scale + offset.
     */
    struct ChannelTransform {
        float scale[3];
        float offset[3];
        bool swapRB;
    };

    ChannelTransform MakeTransform(const ContractMetadata& contract)
    {
        ChannelTransform t{};
        t.swapRB = contract.ConvertBgrToRgb();

        if (contract.NormalizationInGraph()) {
            // The graph applies mean/std normalization internally. 
            // The host only scales from [0, 255] to [0.0, 1.0].
            for (int c = 0; c < 3; ++c) {
                t.scale[c] = 1.0f / 255.0f;
                t.offset[c] = 0.0f;
            }
        }
        else {
            // Host fallback, ImageNet standard: (u8/255 - mean) / std.
            // Fused algebraically into a single multiply-add operation per channel.
            static constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
            static constexpr float stdv[3] = { 0.229f, 0.224f, 0.225f };
            for (int c = 0; c < 3; ++c) {
                t.scale[c] = 1.0f / (255.0f * stdv[c]);
                t.offset[c] = -mean[c] / stdv[c];
            }
        }
        return t;
    }

    /**
     * @brief Converts 8-bit interleaved BGR to 3 planar float channels (NCHW).
     * @details Executed in a single pass over the pixel memory. Optimized with
     * direct pointer increments for maximum auto-vectorization throughput.
     */
    void PackNchw(const cv::Mat& src, float* dst, const ChannelTransform& t)
    {
        const int H = src.rows;
        const int W = src.cols;
        const std::size_t plane = static_cast<std::size_t>(H) * W;

        float* p0 = dst;
        float* p1 = dst + plane;
        float* p2 = dst + 2 * plane;

        const int i0 = t.swapRB ? 2 : 0;
        const int i2 = t.swapRB ? 0 : 2;

        for (int y = 0; y < H; ++y) {
            const uint8_t* row = src.ptr<uint8_t>(y);
            // HPC Tip: Continuous pointer increments (*p++) bypass the need to 
            // recalculate (base + x) offsets on every single pixel iteration.
            for (int x = 0; x < W; ++x, row += 3) {
                *p0++ = row[i0] * t.scale[0] + t.offset[0];
                *p1++ = row[1] * t.scale[1] + t.offset[1];
                *p2++ = row[i2] * t.scale[2] + t.offset[2];
            }
        }
    }

} // namespace


void PrepStage(const AppConfig& cfg,
    const OrtSessionConfig& session,
    RingBuffer<RawFrame*>& qRaw,
    RingBuffer<RawFrame*>& rawPool,
    RingBuffer<PipelineSlot*>& slotPool,
    RingBuffer<PipelineSlot*>& qPrep,
    PerformanceMetrics& metrics)
{
    if (cfg.PrepRealtime()) {
        const DWORD err = RT::ConfigureRealtimeThread();
        if (err != 0)
            Log::Warning("prep: real-time priority not applied (GetLastError={})", err);
    }

    const PatchGeometry& g = cfg.Geometry();
    const ChannelTransform transform = MakeTransform(session.Contract());

    // All scratch memory for this thread, allocated exactly once.
    const AntialiasResizer resizer(static_cast<int>(g.stripWidth),
        static_cast<int>(g.patchHeight),
        session.ModelWidth(), session.ModelHeight());
    cv::Mat scratch = resizer.MakeScratch();
    cv::Mat dest = resizer.MakeDestination();

    // Dummy run (Warmup): Forces the resolution of lazy thread_local allocations 
    // inside the resizer and OpenCV now, preventing jitter on the first real batch.
    {
        cv::Mat warm(static_cast<int>(g.patchHeight), static_cast<int>(g.stripWidth), CV_8UC3, cv::Scalar::all(0));
        cv::Mat wd = dest, ws = scratch;
        resizer.Resize(warm, wd, ws);
    }

    const std::size_t imgElems = session.InputElems() / session.BatchSize();
    const int batch = session.BatchSize();

    for (;;) {
        RawFrame* raw = nullptr;
        if (!qRaw.pop(raw)) break; // Queue stopped: exit thread cleanly

        PipelineSlot* slot = nullptr;
        if (!slotPool.pop(slot)) {
            (void)rawPool.try_push(raw);
            break;
        }

        const auto t0 = std::chrono::steady_clock::now();

        const int n = (std::min)(batch, static_cast<int>(raw->patches.size()));
        for (int i = 0; i < n; ++i) {
            cv::Mat out = dest; // Zero-cost if resizer bypasses due to identity match
            resizer.Resize(raw->patches[i], out, scratch);
            PackNchw(out, slot->input + static_cast<std::size_t>(i) * imgElems, transform);
        }

        slot->seq = raw->seq;
        slot->acquiredQPC = raw->acquiredQPC;
        slot->prepMs = MsSince(t0);

        // Returned immediately upon transformation. The patches point directly 
        // into the raw strip, so every millisecond held longer forces the producer to drop frames.
        (void)rawPool.try_push(raw);

        metrics.addPreprocessingTime(slot->prepMs);

        // If pipeline is shutting down, return the slot to prevent leaks
        if (!qPrep.try_push(slot)) {
            (void)slotPool.try_push(slot);
            break;
        }
    }
}


void InferStage(OrtSessionConfig& session,
    RingBuffer<PipelineSlot*>& qPrep,
    RingBuffer<PipelineSlot*>& qInf,
    RingBuffer<PipelineSlot*>& slotPool,
    PerformanceMetrics& metrics)
{
    // Inference usually sits at the critical path; Real-Time elevation is highly recommended.
    const DWORD err = RT::ConfigureRealtimeThread();
    if (err != 0)
        Log::Warning("prep: real-time priority not applied (GetLastError={})", err);

    for (;;) {
        PipelineSlot* slot = nullptr;
        if (!qPrep.pop(slot)) break;

        OrtSessionConfig::Timings t{};
        try {
            session.RunBatch(slot->input, slot->scores, slot->rawMap, t);
        }
        catch (const std::exception& e) {
            Log::Error("Inference failed on batch {}: {}", slot->seq, e.what());
            (void)slotPool.try_push(slot); // Prevent slot leak on failure
            continue;
        }

        slot->h2dMs = t.h2dMs;
        slot->runMs = t.runMs;
        slot->d2hMs = t.d2hMs;

        metrics.addH2DTime(t.h2dMs);
        metrics.addRunTime(t.runMs);
        metrics.addD2HTime(t.d2hMs);
        metrics.addGpuTime(t.h2dMs + t.runMs + t.d2hMs);

        if (!qInf.try_push(slot)) {
            (void)slotPool.try_push(slot);
            break;
        }
    }
}


void PostStage(const AppConfig& cfg,
    const OrtSessionConfig& session,
    RingBuffer<PipelineSlot*>& qInf,
    RingBuffer<PipelineSlot*>& slotPool,
    IResultSink& sink,
    PerformanceMetrics& metrics)
{
    if (cfg.PrepRealtime()) {
        const DWORD err = RT::ConfigureRealtimeThread();
        if (err != 0)
            Log::Warning("prep: real-time priority not applied (GetLastError={})", err);
    }

    const ContractMetadata& contract = session.Contract();
    const int batch = session.BatchSize();
    const int mapH = session.MapHeight();
    const int mapW = session.MapWidth();
    const std::size_t rawElems = static_cast<std::size_t>(mapH) * mapW;

    const int outH = cfg.MapAtModelResolution() ? mapH : static_cast<int>(cfg.Geometry().patchHeight);
    const int outW = cfg.MapAtModelResolution() ? mapW : static_cast<int>(cfg.Geometry().stripWidth);
    const std::size_t outElems = static_cast<std::size_t>(outH) * outW;

    // Precalculated Min-Max normalization bounds based on model contract.
    // Maps the raw logit outputs from [mapMin, mapMax] to [0, 255] uint8_t.
    const float span = contract.MapMax() - contract.MapMin();
    const double alpha = 255.0 / span;
    const double beta = -255.0 * contract.MapMin() / span;

    // Thread-local scratch buffers, allocated strictly once.
    cv::Mat scaled(mapH, mapW, CV_8UC1);
    cv::Mat resized;
    if (!cfg.MapAtModelResolution()) {
        resized.create(outH, outW, CV_8UC1);
    }

    for (;;) {
        PipelineSlot* slot = nullptr;
        if (!qInf.pop(slot)) break;

        SlotGuard guard(slotPool, slot); // Guaranteed release back to slotPool
        const auto t0 = std::chrono::steady_clock::now();

        int anomalies = 0;
        for (int b = 0; b < batch; ++b) {
            if (slot->scores[b] >= contract.ScoreThreshold()) {
                ++anomalies;
            }

            // Zero-copy wrap around the D2H pinned memory
            const cv::Mat raw(mapH, mapW, CV_32FC1, slot->rawMap + b * rawElems);

            // Fused scale and offset using AVX2-accelerated convertTo
            raw.convertTo(scaled, CV_8UC1, alpha, beta);

            std::uint8_t* outMap = slot->outMap + b * outElems;
            if (cfg.MapAtModelResolution()) {
                std::memcpy(outMap, scaled.data, outElems);
            }
            else {
                cv::Mat dst(outH, outW, CV_8UC1, outMap);
                cv::resize(scaled, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
            }

            if (slot->outMask) {
                cv::Mat src(outH, outW, CV_8UC1, outMap);
                cv::Mat mask(outH, outW, CV_8UC1, slot->outMask + b * outElems);
                const double thr = (contract.PixelThreshold() - contract.MapMin()) * alpha + beta;
                cv::compare(src, thr, mask, cv::CMP_GE); // Fast vectorized binary thresholding
            }
        }

        BatchResult result;
        result.seq = slot->seq;
        result.acquiredQPC = slot->acquiredQPC;
        result.batch = batch;
        result.scores = slot->scores;
        result.map = slot->outMap;
        result.mask = slot->outMask;
        result.mapElemsPerImage = outElems;
        result.anomalies = anomalies;

        sink.Publish(result); // Raw pointers inside `result` are only valid during this call

        slot->postMs = MsSince(t0);
        metrics.addPostprocessingTime(slot->postMs); // Internally increments completedBatches counter
    }
}