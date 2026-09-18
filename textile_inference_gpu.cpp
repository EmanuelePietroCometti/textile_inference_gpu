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
            static_cast<std::uint64_t>(s.completedBatches) - lastBatches,
            s.preprocessing, s.h2d, s.run, s.d2h, s.postprocessing, total,
            source.FramesRead(), source.DroppedNoBuffer());
    }

    /**
     * @brief Raw, non-blocking keyboard reader on the console input buffer.
     *
     * @details Waits on the stdin handle with a timeout, so the same loop can also
     * drive the metrics cadence. Line buffering and echo are disabled (single key, no
     * Enter). QuickEdit is disabled too: with QuickEdit on, a mouse click in the console
     * freezes stdout, the logger thread blocks inside fmt::print and log records start
     * being dropped. The original console mode is restored by the destructor.
     */
    class ConsoleKeys {
    public:
        ConsoleKeys()
        {
            h_ = GetStdHandle(STD_INPUT_HANDLE);
            if (h_ == nullptr || h_ == INVALID_HANDLE_VALUE || !GetConsoleMode(h_, &oldMode_))
                throw std::runtime_error("stdin is not an interactive console: keyboard control unavailable");

            const DWORD mode = (oldMode_ & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
                | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT | ENABLE_QUICK_EDIT_MODE))
                | ENABLE_EXTENDED_FLAGS;
            if (!SetConsoleMode(h_, mode))
                throw std::runtime_error("SetConsoleMode failed, GetLastError=" + std::to_string(GetLastError()));

            FlushConsoleInputBuffer(h_); // ignore keys typed during the (long) initialization
        }

        ~ConsoleKeys() { SetConsoleMode(h_, oldMode_); }

        ConsoleKeys(const ConsoleKeys&) = delete;
        ConsoleKeys& operator=(const ConsoleKeys&) = delete;

        /// @return The lowercase character of the next key press, or 0 on timeout / non-key events.
        wchar_t Poll(DWORD timeoutMs)
        {
            if (WaitForSingleObject(h_, timeoutMs) != WAIT_OBJECT_0) return 0;

            // One record per call: if more are pending, the next Poll returns immediately,
            // so no key is lost when several events are queued at once.
            INPUT_RECORD rec{};
            DWORD n = 0;
            if (!ReadConsoleInputW(h_, &rec, 1, &n) || n == 0) return 0;
            if (rec.EventType != KEY_EVENT || !rec.Event.KeyEvent.bKeyDown) return 0;

            return static_cast<wchar_t>(std::towlower(rec.Event.KeyEvent.uChar.UnicodeChar));
        }

    private:
        HANDLE h_ = nullptr;
        DWORD oldMode_ = 0;
    };

    /**
     * @brief Owns the ingest thread: one thread per listening session (s ... x).
     *
     * @details Start() re-arms the MMF source and spawns the thread; Stop() signals the
     * source and joins it. The stage threads are NOT touched: while not listening they
     * simply block on an empty qRaw at zero CPU cost.
     */
    class IngestController {
    public:
        IngestController(MmfFrameSource& source, std::uint32_t readTimeoutMs)
            : source_(source), readTimeoutMs_(readTimeoutMs) {
        }

        ~IngestController() { Stop(); }

        IngestController(const IngestController&) = delete;
        IngestController& operator=(const IngestController&) = delete;

        bool Listening() const { return thread_.joinable(); }

        /// False if the thread exited on its own (e.g. WaitForSingleObject failure) while still "listening".
        bool Alive() const { return alive_.load(std::memory_order_acquire); }

        void Start()
        {
            if (thread_.joinable()) return;

            source_.Start();
            alive_.store(true, std::memory_order_release);
            thread_ = std::thread([this] {
                const DWORD err = RT::ConfigureRealtimeThread();
                if (err != 0)
                    Log::Warning("ingest: real-time priority not applied (GetLastError={})", err);

                while (source_.ReadFrame(readTimeoutMs_) != FrameStatus::Stopped) {}

                alive_.store(false, std::memory_order_release);
                });
        }

        void Stop()
        {
            if (!thread_.joinable()) return;
            source_.Stop();
            thread_.join();
        }

    private:
        MmfFrameSource& source_;
        const std::uint32_t readTimeoutMs_;
        std::thread thread_;
        std::atomic<bool> alive_{ false };
    };

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
            RingBuffer<PipelineSlot*>& slotPool)
            : qRaw_(qRaw), rawPool_(rawPool), qPrep_(qPrep), qInf_(qInf), slotPool_(slotPool) {
        }

        ~StageWorkers() { Abort(); }

        StageWorkers(const StageWorkers&) = delete;
        StageWorkers& operator=(const StageWorkers&) = delete;

        std::vector<std::thread> prep, infer, post;

        void Drain()
        {
            qRaw_.stop();  JoinAll(prep);
            qPrep_.stop(); JoinAll(infer);
            qInf_.stop();  JoinAll(post);
            // Pools are stopped last: during the drain prep may still be waiting for a
            // slot that a post thread is about to give back.
            slotPool_.stop();
            rawPool_.stop();
        }

        void Abort()
        {
            qRaw_.stop(); qPrep_.stop(); qInf_.stop();
            slotPool_.stop(); rawPool_.stop();
            JoinAll(prep); JoinAll(infer); JoinAll(post);
        }

    private:
        static void JoinAll(std::vector<std::thread>& ts)
        {
            for (auto& t : ts) if (t.joinable()) t.join();
        }

        RingBuffer<RawFrame*>& qRaw_;
        RingBuffer<RawFrame*>& rawPool_;
        RingBuffer<PipelineSlot*>& qPrep_;
        RingBuffer<PipelineSlot*>& qInf_;
        RingBuffer<PipelineSlot*>& slotPool_;
    };

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

        for (std::uint32_t i = 0; i < cfg.PrepThreads(); ++i) {
            workers.prep.emplace_back(PrepStage, std::cref(cfg), std::cref(proto),
                std::ref(qRaw), std::ref(rawStore.Pool()),
                std::ref(slotStore.Pool()), std::ref(qPrep), std::ref(metrics));
        }
        for (std::uint32_t i = 0; i < cfg.InferenceThreads(); ++i) {
            // Captures by reference are safe: everything referenced is declared before
            // `workers`, which joins this thread before any of it is destroyed.
            workers.infer.emplace_back([&, i] {
            // Inference sits on the critical path: real-time elevation recommended.
            if (cfg.InferenceRealtime()) {
                const DWORD err = RT::ConfigureRealtimeThread();
                if (err != 0)
                    Log::Warning("infer: real-time priority not applied (GetLastError={})", err);
                }
                InferStage(*sessions[i], qPrep, qInf, slotStore.Pool(), metrics);
            });
        }
        for (std::uint32_t i = 0; i < cfg.PostThreads(); ++i) {
            workers.post.emplace_back(PostStage, std::cref(cfg), std::cref(proto),
                std::ref(qInf), std::ref(slotStore.Pool()),
                std::ref(sink), std::ref(metrics));
        }

        // Declared last => destroyed first: the ingest thread stops before anything else.
        IngestController ingest(source, cfg.ReadTimeoutMs());

        Log::Info("Pipeline ready: {} Prep, {} Infer, {} Post threads.",
            cfg.PrepThreads(), cfg.InferenceThreads(), cfg.PostThreads());
        Log::Info("Commands: [s] start listening on MMF | [x] stop listening | [q] quit");

        // Main Thread: keyboard + telemetry loop.
        // The key poll timeout (100 ms) replaces the old sleep and sets the telemetry granularity.
        const auto printEvery = std::chrono::milliseconds(cfg.MetricsPrintEveryMs());
        std::uint64_t lastBatches = 0;
        auto nextPrint = std::chrono::steady_clock::now();
        bool quit = false;
        while(!quit) {
            switch (keys.Poll(0)) {
            case L's':
                if (ingest.Listening()) {
                    Log::Info("Already listening on the MMF");
                    break;
                }
                ingest.Start();
                lastBatches = static_cast<std::uint64_t>(metrics.snapshot().completedBatches);
                nextPrint = std::chrono::steady_clock::now() + printEvery;
                Log::Info("Listening on the MMF: inference started");
                break;

            case L'x':
                if (!ingest.Listening()) {
                    Log::Info("Not listening: nothing to stop");
                    break;
                }
                ingest.Stop();
                // Frames already in the pipeline complete normally: x stops the input, not the work.
                PrintMetrics(metrics, source, lastBatches);
                lastBatches = static_cast<std::uint64_t>(metrics.snapshot().completedBatches);
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

            // Telemetry only while listening: when idle the numbers would never change.
            if (!ingest.Listening() || cfg.MetricsPrintEveryMs() == 0) continue;

            const auto now = std::chrono::steady_clock::now();
            if (now < nextPrint) continue;

            nextPrint = now + printEvery;
            PrintMetrics(metrics, source, lastBatches);
            lastBatches = static_cast<std::uint64_t>(metrics.snapshot().completedBatches);
        }

        // Graceful Shutdown (Execution Order is Critical):
        // 1. input off  2. drain the stages in pipeline order  3. the destructors
        //    release the rest in reverse declaration order (source -> MMF handles,
        //    SlotStore -> pinned memory, sessions -> ORT/TensorRT, RealTimeScope,
        //    console mode, logger last).
        Log::Info("Shutdown initiated...");
        ingest.Stop();
        workers.Drain();

        PrintMetrics(metrics, source, lastBatches);
        Log::Info("Terminated | total batches {} | frames read {} | dropped {} | lost log messages {}",
            metrics.snapshot().completedBatches, source.FramesRead(),
            source.DroppedNoBuffer(), logger.DroppedCount());
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
