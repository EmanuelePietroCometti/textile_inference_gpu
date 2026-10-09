#include "IngestController.h"

#include "AsyncLogger.h"
#include "MmfFrameSource.h"
#include "RealTimeConfig.h"

IngestController::IngestController(MmfFrameSource& source, std::uint32_t readTimeoutMs, int priority)
    : source_(source), readTimeoutMs_(readTimeoutMs), priority_(priority) {
}

IngestController::~IngestController() { Stop(); }

void IngestController::Start()
{
    if (thread_.joinable()) return;

    source_.Start();
    alive_.store(true, std::memory_order_release);
	listenStart_ = std::chrono::high_resolution_clock::now();
    thread_ = std::thread([this] {
        if (const DWORD err = RT::ConfigureThread(priority_); err != 0)
            Log::Warning("ingest: thread priority {} not applied (GetLastError={})", priority_, err);

        while (source_.ReadFrame(readTimeoutMs_) != FrameStatus::Stopped) {}

        alive_.store(false, std::memory_order_release);
        });
}

void IngestController::Stop()
{
    if (!thread_.joinable()) return;
    source_.Stop();
    thread_.join();
    
	listenedSec_ = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - listenStart_).count();
}

double IngestController::ListenedSec()
{
    if (thread_.joinable()) {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - listenStart_).count();
	}
    return listenedSec_;
}
