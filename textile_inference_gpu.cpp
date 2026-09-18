#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "AppConfig.h"
#include "AsyncLogger.h"
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

namespace {

    std::atomic<bool> g_running{ true };

    BOOL WINAPI ConsoleHandler(DWORD signal)
    {
        if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT || signal == CTRL_BREAK_EVENT) {
            g_running.store(false, std::memory_order_relaxed);
            return TRUE;
        }
        return FALSE;
    }

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

    void PrintMetrics(const PerformanceMetrics& metrics,
        const MmfFrameSource& source, std::uint64_t lastBatches)
    {
        const auto s = metrics.snapshot();
        const double total = s.preprocessing + s.gpu + s.postprocessing;

        Log::Info("Metrics | batch {} | prep {:.2f} | h2d {:.2f} run {:.2f} d2h {:.2f} | post {:.2f} | total {:.2f} ms | read {} dropped {}",
            s.completedBatches - lastBatches,
            s.preprocessing, s.h2d, s.run, s.d2h, s.postprocessing, total,
            source.FramesRead(), source.DroppedNoBuffer());
    }

} // namespace


int main(int argc, char** argv)
{
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    try {
        // Configuration & Logger Setup
        const auto iniPath = ResolveIniPath(argc, argv);
        const AppConfig cfg = AppConfig::LoadFromIni(iniPath.wstring());

        AsyncLogger logger(cfg.LogMaxMessageChars(), cfg.LogQueueCapacity(),
            cfg.LogFlushIntervalMs(), cfg.LogNotifyThreshold());
        logger.Start();

        // The configuration dump MUST happen AFTER the logger starts, otherwise 
        // Log::Info is a no-op and crucial diagnostic data is lost.
        Log::Info("INI Path: {}", iniPath.string());
        Log::Info("{}", cfg.Describe());

        // Elevate process priority for the entire duration of the application.
        // RAII guarantees restoration even if an exception causes an early exit.
        std::unique_ptr<RealTimeScope> rtScope;
        if (cfg.ElevateProcess()) {
            rtScope = std::make_unique<RealTimeScope>();
            Log::Info("Process priority elevated: realtime={} elevated={}",
                rtScope->realtime(), rtScope->elevated());
        }

        // Prep and Post threads execute cv::parallel_for_ internally.
        // These counts multiply. AppConfig guarantees this product doesn't exceed 
        // logical processor limits to prevent OS context-switch thrashing.
        cv::setNumThreads(static_cast<int>(cfg.OpenCvThreads()));

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

        SlotStore slotStore(proto, cfg.Slots(), outMapElems, cfg.DrawMask());

        RingBuffer<PipelineSlot*> qPrep(cfg.QPrepCapacity());
        RingBuffer<PipelineSlot*> qInf(cfg.QInfCapacity());

        // Cross-module validation: A check no single module can perform alone.
        if (proto.ModelChannels() != static_cast<int>(cfg.Geometry().channels)) {
            throw std::runtime_error("Channel mismatch: IPC source provides "
                + std::to_string(cfg.Geometry().channels) + ", but model expects "
                + std::to_string(proto.ModelChannels()));
        }
        if (proto.BatchSize() != static_cast<int>(cfg.Geometry().count)) {
            throw std::runtime_error("Model batch size (" + std::to_string(proto.BatchSize())
                + ") differs from required patch count (" + std::to_string(cfg.Geometry().count) + ")");
        }

        PerformanceMetrics metrics(cfg.MetricsWindow(), proto.BatchSize());
        LoggingResultSink sink;

        MmfFrameSource source(layout, rawStore.Pool(), qRaw);

        // TensorRT Warmup
        // For TensorRT, the first execution triggers JIT compilation or deserializes 
        // the engine. Doing this BEFORE opening the IPC channel ensures the first 
        // real frames are processed instantly rather than dropped due to timeouts.
        for (auto& s : sessions) {
            s->Warmup(cfg.Model().WarmupRuns());
        }

        // Thread Pool Spawning
        std::thread ingest([&] {
            const DWORD err = RT::ConfigureRealtimeThread();
            if (err != 0)
                Log::Warning("prep: real-time priority not applied (GetLastError={})", err);
            while (g_running.load(std::memory_order_relaxed)) {
                if (source.ReadFrame(cfg.ReadTimeoutMs()) == FrameStatus::Stopped) break;
            }
            Log::Info("Ingest thread terminated | read {} | dropped due to empty pool {}",
                source.FramesRead(), source.DroppedNoBuffer());
            });

        std::vector<std::thread> prep, infer, post;
        for (std::uint32_t i = 0; i < cfg.PrepThreads(); ++i) {
            prep.emplace_back(PrepStage, std::cref(cfg), std::cref(proto),
                std::ref(qRaw), std::ref(rawStore.Pool()),
                std::ref(slotStore.Pool()), std::ref(qPrep), std::ref(metrics));
        }

        for (std::uint32_t i = 0; i < cfg.InferenceThreads(); ++i) {
            infer.emplace_back(InferStage, std::ref(*sessions[i]),
                std::ref(qPrep), std::ref(qInf), std::ref(slotStore.Pool()),
                std::ref(metrics));
        }

        for (std::uint32_t i = 0; i < cfg.PostThreads(); ++i) {
            post.emplace_back(PostStage, std::cref(cfg), std::cref(proto),
                std::ref(qInf), std::ref(slotStore.Pool()),
                std::ref(sink), std::ref(metrics));
        }

        Log::Info("Pipeline running: {} Prep, {} Infer, {} Post threads. Press Ctrl+C to terminate.",
            cfg.PrepThreads(), cfg.InferenceThreads(), cfg.PostThreads());

        // Main Thread Telemetry Loop
        // Delegating telemetry to the main thread prevents multiple PostThreads 
        // from interlacing log lines and distorting the timing cadence.
        std::uint64_t lastBatches = 0;
        auto nextPrint = std::chrono::steady_clock::now();

        while (g_running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (cfg.MetricsPrintEveryMs() == 0) continue;

            const auto now = std::chrono::steady_clock::now();
            if (now < nextPrint) continue;

            nextPrint = now + std::chrono::milliseconds(cfg.MetricsPrintEveryMs());
            PrintMetrics(metrics, source, lastBatches);
            lastBatches = metrics.snapshot().completedBatches;
        }

        // Graceful Shutdown (Execution Order is Critical)
        // Threads must be joined BEFORE their respective memory stores/pools 
        // go out of scope. E.g., Prep threads hold cv::Mat headers that point 
        // directly into the rawStore buffers. Destroying the pool first = Segfault.
        Log::Info("Shutdown initiated...");

        source.Stop();
        ingest.join();

        qRaw.stop();
        for (auto& t : prep) t.join();

        qPrep.stop();
        for (auto& t : infer) t.join();

        qInf.stop();
        for (auto& t : post) t.join();

        slotStore.Pool().stop();
        rawStore.Pool().stop();

        PrintMetrics(metrics, source, lastBatches);
        Log::Info("Terminated | total batches {} | lost log messages {}",
            metrics.snapshot().completedBatches, logger.DroppedCount());

        logger.Stop();
        return 0;
    }
    catch (const std::exception& e) {
        // If initialization fails early, the async logger might not exist or 
        // might already be stopped. Standard error guarantees visibility.
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        Log::Error("Fatal Error: {}", e.what());
        return -1;
    }
}