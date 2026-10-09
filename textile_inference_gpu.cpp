#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <cwctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "AppConfig.h"
#include "AsyncLogger.h"
#include "ConsoleKeys.h"
#include "IngestController.h"
#include "MmfFrameSource.h"
#include "OrtSession.h"
#include "PatchLayout.h"
#include "PerformanceMetrics.h"
#include "PipelineSlot.h"
#include "RawFrame.h"
#include "RealTimeConfig.h"
#include "ResultSink.h"
#include "RingBuffer.h"
#include "Stages.h"
#include "StageWorkers.h"

namespace {

    /**
     * @brief Resolves the INI configuration path relative to the executable, not the CWD.
     * @details Launching from Visual Studio or a background service often results in a
     * working directory that differs from the binary location. argv[1] allows overriding.
     */
    std::filesystem::path ResolveIniPath(int argc, char** argv)
    {
        if (argc > 1) return std::filesystem::path(argv[1]);

        wchar_t buffer[MAX_PATH]{};
        GetModuleFileNameW(nullptr, buffer, MAX_PATH);
        return std::filesystem::path(buffer).parent_path() / L"iniConfigFile_onnxInference.ini";
    }

    int Run(int argc, char** argv)
    {
        // Configuration & Logger Setup
        const auto iniPath = ResolveIniPath(argc, argv);
        const AppConfig cfg = AppConfig::LoadFromIni(iniPath.wstring());

        // Declared first => destroyed last: the destructors below (RealTimeScope restore,
        // StageWorkers, SlotStore...) can still log. No explicit logger.Stop() needed.
        AsyncLogger logger(cfg.LogMaxMessageChars(), cfg.LogQueueCapacity(),
            cfg.LogFlushIntervalMs(), cfg.LogNotifyThreshold());
        logger.Start();

        // The configuration dump MUST happen AFTER the logger starts, otherwise
        // Log::Info is a no-op and crucial diagnostic data is lost.
        Log::Info("INI Path: {}", iniPath.string());
        Log::Info("{}", cfg.Describe());

        // Elevate process priority for the entire duration of the application.
        std::unique_ptr<RealTimeScope> rtScope;
        if (cfg.ElevateProcess()) {
            rtScope = std::make_unique<RealTimeScope>();
            Log::Info("Process priority elevated: realtime={} elevated={}",
                rtScope->realtime(), rtScope->elevated());
        }

        // Prep and Post threads execute cv::parallel_for_ internally: the counts multiply.
        cv::setNumThreads(static_cast<int>(cfg.OpenCvThreads()));

        // CUDA host-wait policy. Must run before any allocation that touches the context
        // (pinned stores, sessions). With the default policy (cudaDeviceScheduleAuto, one
        // context, many cores) every cudaStreamSynchronize SPINS: the inference threads,
        // at TIME_CRITICAL, would burn a core each for the whole GPU run, stealing it from
        // prep, post and the producer process. BlockingSync makes our waits and the ones
        // inside ORT/TensorRT sleep on an OS event; the wake-up costs microseconds against
        // a batch of tens of milliseconds.
        {
            const int dev = cfg.Model().DeviceId();
            if (const cudaError_t e = cudaSetDevice(dev); e != cudaSuccess)
                throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(e));
            if (const cudaError_t e = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); e != cudaSuccess)
                Log::Warning("cudaSetDeviceFlags(BlockingSync) failed: {}. Host waits will spin.",
                    cudaGetErrorString(e));
            unsigned int flags = 0;
            cudaGetDeviceFlags(&flags);
            Log::Info("CUDA device {} | flags 0x{:x} | blocking sync {}", dev, flags,
                (flags & cudaDeviceScheduleMask) == cudaDeviceScheduleBlockingSync ? "ON" : "OFF");
        }

        // Bottom-Up Architecture Initialization
        const PatchLayout layout(cfg.Geometry());

        RawFrameStore rawStore(layout, cfg.RawFrames());
        RingBuffer<RawFrame*> qRaw(cfg.QRawCapacity());

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "textile_inference_gpu");

        std::vector<std::unique_ptr<OrtSessionConfig>> sessions;
        sessions.reserve(cfg.InferenceThreads());
        for (std::uint32_t i = 0; i < cfg.InferenceThreads(); ++i) {
            sessions.push_back(std::make_unique<OrtSessionConfig>(env, cfg.Model()));
        }

        const OrtSessionConfig& proto = *sessions.front();

        // The output map geometry is dictated strictly by the ONNX model graph.
        const std::size_t outMapElems = cfg.MapAtModelResolution()
            ? static_cast<std::size_t>(proto.MapHeight()) * proto.MapWidth()
            : static_cast<std::size_t>(cfg.Geometry().patchHeight) * cfg.Geometry().stripWidth;

        // Cross-module validation: checks no single module can perform alone.
        // Done BEFORE SlotStore, so a mismatch does not pin hundreds of MiB just to throw.
        if (proto.ModelChannels() != static_cast<int>(cfg.Geometry().channels)) {
            throw std::runtime_error("Channel mismatch: IPC source provides "
                + std::to_string(cfg.Geometry().channels) + ", but model expects "
                + std::to_string(proto.ModelChannels()));
        }
        if (proto.BatchSize() != static_cast<int>(cfg.Geometry().count)) {
            throw std::runtime_error("Model batch size (" + std::to_string(proto.BatchSize())
                + ") differs from required patch count (" + std::to_string(cfg.Geometry().count) + ")");
        }
        // PrepStage supports two geometries only: native (model == patch) and the fused
        // exact 2x downscale (model == patch / 2). Anything else needs a general resampler.
        {
            const int ph = static_cast<int>(cfg.Geometry().patchHeight);
            const int pw = static_cast<int>(cfg.Geometry().stripWidth);
            const int mh = proto.ModelHeight();
            const int mw = proto.ModelWidth();
            const bool native = (mh == ph && mw == pw);
            const bool half = (2 * mh == ph && 2 * mw == pw);
            if (!native && !half) {
                throw std::runtime_error("Model input " + std::to_string(mh) + "x" + std::to_string(mw)
                    + " vs patch " + std::to_string(ph) + "x" + std::to_string(pw)
                    + ": supported ratios are 1:1 and 2:1");
            }
            Log::Info("Preprocessing: patch {}x{} -> model {}x{} | {}", ph, pw, mh, mw,
                native ? "no resize"
                : (cfg.ResizeAntialias() ? "2x antialias (triangle, = old AntialiasResizer)"
                    : "2x box (= cv::resize INTER_LINEAR)"));
        }

		// The SlotStore allocates all the pinned memory for the entire pipeline, so it must be created after the model geometry is known. The store is shared
		// between the prep, infer, and post threads, which is why it is declared here in main() and passed by reference to each stage.
        SlotStore slotStore(proto, cfg.Slots(), outMapElems, cfg.DrawMask());

		// The queues are the only inter-stage communication mechanism. They are declared after the stores they reference, so the StageWorkers destructor joins the threads before the stores are destroyed.
        RingBuffer<PipelineSlot*> qPrep(cfg.QPrepCapacity());
        RingBuffer<PipelineSlot*> qInf(cfg.QInfCapacity());

		// Telemetry: moving averages of the last N batches, plus a total batch counter.
        PerformanceMetrics metrics(cfg.MetricsWindow());
        LoggingResultSink sink;

		// The MMF source is the only producer of frames: it reads from the shared memory and pushes them into qRaw. It is declared after rawStore and qRaw, so it can be destroyed before them.
        MmfFrameSource source(layout, rawStore.Pool(), qRaw);

        // TensorRT Warmup: engine build/deserialization happens here, before anyone
        // can press 's', so the first real frames are not dropped.
        for (auto& s : sessions) {
            s->Warmup(cfg.Model().WarmupRuns());
        }

        // Fail fast if there is no console, before spawning any thread.
        ConsoleKeys keys;

        // Stage threads: spawned once, alive until 'q'. Declared AFTER every store and
        // queue they reference, so they are joined before those are destroyed, on the
        // normal path and on the exception path alike.
        StageWorkers workers(qRaw, rawStore.Pool(), qPrep, qInf, slotStore.Pool());

		// Prep threads: mostly CPU-bound, so they run at TIME_CRITICAL to preempt the GPU threads and keep the GPU busy. The PrepStage function is reentrant, so multiple threads can run it concurrently. 
        for (std::uint32_t i = 0; i < cfg.PrepThreads(); ++i) {
            workers.prep.emplace_back(PrepStage, std::cref(cfg), std::cref(proto),
                std::ref(qRaw), std::ref(rawStore.Pool()),
                std::ref(slotStore.Pool()), std::ref(qPrep), std::ref(metrics));
        }
        for (std::uint32_t i = 0; i < cfg.InferenceThreads(); ++i) {
            // Captures by reference are safe: everything referenced is declared before
            // `workers`, which joins this thread before any of it is destroyed.
            workers.infer.emplace_back([&, i] {
                // Mostly asleep on the GPU (blocking sync): short CPU bursts to enqueue
                // work, which must preempt prep/post so the GPU never waits for the host.
                if (const DWORD err = RT::ConfigureThread(cfg.InferencePriority()); err != 0)
                    Log::Warning("infer: thread priority {} not applied (GetLastError={})",
                        cfg.InferencePriority(), err);
                InferStage(*sessions[i], qPrep, qInf, slotStore.Pool(), metrics);
            });
        }

		// Post threads: mostly CPU-bound, so they run at TIME_CRITICAL to preempt the GPU threads and keep the GPU busy. The PostStage function is reentrant, so multiple threads can run it concurrently.
        for (std::uint32_t i = 0; i < cfg.PostThreads(); ++i) {
            workers.post.emplace_back(PostStage, std::cref(cfg), std::cref(proto),
                std::ref(qInf), std::ref(slotStore.Pool()),
                std::ref(sink), std::ref(metrics));
        }

        // Declared last => destroyed first: the ingest thread stops before anything else.
        IngestController ingest(source, cfg.ReadTimeoutMs(), cfg.IngestPriority());

        Log::Info("Pipeline ready: {} Prep, {} Infer, {} Post threads.",
            cfg.PrepThreads(), cfg.InferenceThreads(), cfg.PostThreads());
        Log::Info("Commands: [s] start listening on MMF | [x] stop listening | [q] quit");

        // Main Thread: keyboard loop. Metrics are not printed from here: the Post stage logs one
        // [MONITOR] line every [Metrics] WindowDimension completed batches.
        // Poll() BLOCKS on the console handle for up to kPollMs: zero CPU while idle, and
        // kPollMs is how often the ingest thread liveness is checked. With a 0 timeout this
        // loop spun a whole core forever, at base priority 24 under REALTIME_PRIORITY_CLASS.
        constexpr DWORD kPollMs = 100;
        bool quit = false;
        while (!quit) {
            switch (keys.Poll(kPollMs)) {
            case L's':
                if (ingest.Listening()) {
                    Log::Info("Already listening on the MMF");
                    break;
                }
                ingest.Start();
                Log::Info("Listening on the MMF: inference started");
                break;

            case L'x':
                if (!ingest.Listening()) {
                    Log::Info("Not listening: nothing to stop");
                    break;
                }
                ingest.Stop();
                // Frames already in the pipeline complete normally: x stops the input, not the work.
                Log::Info("Stopped listening on the MMF. [s] to restart, [q] to quit");
                break;

            case L'q':
                quit = true;
                break;

            default:
                break;
            }

            if (ingest.Listening() && !ingest.Alive()) {
                Log::Error("Ingest thread terminated unexpectedly: listening stopped. [s] to retry");
                ingest.Stop(); // joins the (already finished) thread
            }
        }

        // Graceful Shutdown (Execution Order is Critical):
        // 1. input off  2. drain the stages in pipeline order  3. the destructors
        //    release the rest in reverse declaration order (source -> MMF handles,
        //    SlotStore -> pinned memory, sessions -> ORT/TensorRT, RealTimeScope,
        //    console mode, logger last).
        Log::Info("Shutdown initiated...");
        ingest.Stop();
        workers.Drain();

        metrics.logFinalReport({
            .framesRead = source.FramesRead(),
            .framesDropped = source.DroppedNoBuffer(),
            .lostLogMessages = logger.DroppedCount(),
            .listenedSec = ingest.ListenedSec(),
            .prepThreads = cfg.PrepThreads(),
            .inferThreads = cfg.InferenceThreads(),
            .postThreads = cfg.PostThreads()
        });
        return 0;
    }

} // namespace


int main(int argc, char** argv)
{
    try {
        return Run(argc, argv);
    }
    catch (const std::exception& e) {
        // By now Run() has unwound: threads joined, resources released, logger stopped.
        // Standard error guarantees visibility.
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return -1;
    }
}
