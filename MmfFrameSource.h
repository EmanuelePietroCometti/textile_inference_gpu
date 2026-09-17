#pragma once
#include <Windows.h>
#include "SharedFrameContract.h"
#include "RingBuffer.h"
#include "RawFrame.h"

enum class FrameStatus {Ok, TimeOut, Stopped, GeometryMismatch};

class MmfFrameSource {
public:
	MmfFrameSource(const PatchLayout& layout,
		RingBuffer<RawFrame*>& rawPool,
		RingBuffer<RawFrame*>& outQueue);
	~MmfFrameSource();

	MmfFrameSource(const MmfFrameSource&) = delete;
	MmfFrameSource& operator=(const MmfFrameSource&) = delete;

	FrameStatus ReadFrame(std::uint32_t timeoutMs);

	void Stop();

	std::uint64_t FramesRead() const;
	std::uint64_t DroppedNoBuffer() const;
private:
	RingBuffer<RawFrame*>& rawPool_;
	RingBuffer<RawFrame*>& outQueue_;


	HANDLE hMapping;
	void* pView;
	HANDLE hEvent;

	sfc::Frame* frame_ = nullptr;

	std::uint64_t seq_ = 0;

	std::atomic<std::uint64_t> framesRead_{ 0 };
    std::atomic<std::uint64_t> droppedNoBuffer_{ 0 };
    std::atomic<bool> stopped_{ false };
};