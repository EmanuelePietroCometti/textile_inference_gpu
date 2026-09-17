#include "MmfFrameSource.h"
#include "AsyncLogger.h"
#include <memory>

MmfFrameSource::MmfFrameSource(const PatchLayout& layout, RingBuffer<RawFrame*>& rawPool, RingBuffer<RawFrame*>& outQueue) :
	rawPool_(rawPool),
	outQueue_(outQueue),
	hMapping(nullptr),
	pView(nullptr),
	hEvent(nullptr)
{
	if (layout.PayloadBytes() != sfc::FULL_PAYLOAD)
		throw std::runtime_error(
			"INI geometry incompatible with the compiled contract: INI "
			+ std::to_string(layout.PayloadBytes()) + " bytes ("
			+ std::to_string(layout.StripWidth()) + "x"
			+ std::to_string(layout.StripHeight()) + "x"
			+ std::to_string(layout.Channels()) + "), contract "
			+ std::to_string(sfc::FULL_PAYLOAD) + " bytes. "
			"Align the INI with SharedFrameContract.h, or recompile both programs."
		);

	const std::size_t mappingBytes = sizeof(sfc::Frame);

	hMapping = CreateFileMappingW(
		INVALID_HANDLE_VALUE,
		NULL,
		PAGE_READWRITE,
		0,
		static_cast<DWORD>(mappingBytes),
		sfc::kMappingName
	);
	if (!hMapping) {
		throw std::runtime_error("CreateFileMapping failed, GetLastError="+ std::to_string(GetLastError()));
	}
		
	pView = MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	if (!pView) {
		const DWORD err = GetLastError();
		CloseHandle(hMapping); hMapping = nullptr;
		throw std::runtime_error("MapViewOfFile failed, GetLastError=" + std::to_string(err));
	}

	MEMORY_BASIC_INFORMATION mbi{};
	if (VirtualQuery(pView, &mbi, sizeof(mbi)) == 0) {
		const DWORD err = GetLastError();
		UnmapViewOfFile(pView); CloseHandle(hMapping);
		pView = nullptr; hMapping = nullptr;
		throw std::runtime_error("VirtualQuery failed, GetLastError=" + std::to_string(err));
	}

	hEvent = CreateEventW(nullptr, FALSE, FALSE, sfc::kReadyEventName);
	if (!hEvent) {
		const DWORD err = GetLastError();
		UnmapViewOfFile(pView); CloseHandle(hMapping);
		pView = nullptr; hMapping = nullptr;
		throw std::runtime_error("CreateEvent failed, GetLastError=" + std::to_string(err));
	}

	frame_ = static_cast<sfc::Frame*>(pView);

	Log::Info("Shared MMF opened | payload {} byte ({:.1f} MiB) | stripe {}x{}x{}",
		sfc::FULL_PAYLOAD, sfc::FULL_PAYLOAD / (1024.0 * 1024.0),
		layout.StripWidth(), layout.StripHeight(), layout.Channels());
}

MmfFrameSource::~MmfFrameSource()
{
	if (hEvent) CloseHandle(hEvent);
	if (pView) UnmapViewOfFile(pView);
	if (hMapping) CloseHandle(hMapping);
}

FrameStatus MmfFrameSource::ReadFrame(std::uint32_t timeoutMs)
{
	if (stopped_)
	{
		return FrameStatus::Stopped;
	}

	if (InterlockedCompareExchange(&frame_->canRead, 0, 0) == FALSE)
	{
		const DWORD r = WaitForSingleObject(hEvent, timeoutMs);
		if (r == WAIT_TIMEOUT)  return FrameStatus::TimeOut;
		if (r != WAIT_OBJECT_0) return FrameStatus::Stopped;
		if (stopped_.load(std::memory_order_relaxed))
			return FrameStatus::Stopped;
	}

	RawFrame* raw = nullptr;
	if (!rawPool_.try_pop(raw))
	{
		droppedNoBuffer_.fetch_add(1, std::memory_order_relaxed);
		return FrameStatus::TimeOut;
	}

	std::memcpy(raw->strip, frame_->payload_, sfc::FULL_PAYLOAD);

	InterlockedExchange(&frame_->canRead, FALSE);

	LARGE_INTEGER qpc{};
	QueryPerformanceCounter(&qpc);
	raw->acquiredQPC = qpc.QuadPart;
	raw->seq = seq_++;

	framesRead_.fetch_add(1, std::memory_order_relaxed);

	if (!outQueue_.try_push(raw))
	{
		rawPool_.try_push(raw);
		return FrameStatus::Stopped;
	}
	return FrameStatus::Ok;

}

std::uint64_t MmfFrameSource::FramesRead() const
{
	return framesRead_.load(std::memory_order_relaxed);
}

std::uint64_t MmfFrameSource::DroppedNoBuffer() const
{
	return droppedNoBuffer_.load(std::memory_order_relaxed);
}

void MmfFrameSource::Stop()
{
	stopped_.store(true, std::memory_order_relaxed);
}