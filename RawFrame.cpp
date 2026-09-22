#include "RawFrame.h"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
	std::size_t CheckedFrameCount(std::size_t n)
	{
		if (n == 0)
			throw std::invalid_argument("RawFrameStore requires at least one frame");
		return n;
	}
}

RawFrameStore::RawFrameStore(const PatchLayout& layout, std::size_t frameCount) :
	layout_(layout), pool_(CheckedFrameCount(frameCount))
{
	const std::size_t bytes = layout_.PayloadBytes();
	const int cvType = CV_8UC(static_cast<int>(layout_.Channels()));

	pinned_.reserve(frameCount);
	frames_.resize(frameCount);   // before taking addresses: the pool stores &frames_[k]

	try {
		for (std::size_t k = 0; k < frameCount; ++k)
		{
			void* p = nullptr;
			const cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
			if (err != cudaSuccess)
				throw std::runtime_error("RawFrameStore: cudaHostAlloc of " + std::to_string(bytes)
					+ " bytes failed: " + cudaGetErrorString(err));
			pinned_.push_back(p);

			std::uint8_t* strip = static_cast<std::uint8_t*>(p);
			std::memset(strip, 0, bytes);

			RawFrame& f = frames_[k];
			f.strip = strip;
			f.patches.reserve(layout_.Count());

			for (std::uint32_t i = 0; i < layout_.Count(); ++i)
			{
				f.patches.emplace_back(static_cast<int>(layout_.PatchHeight()), static_cast<int>(layout_.StripWidth()), cvType, strip + layout_.Offset(i));
			}

			if (!pool_.try_push(&f))
			{
				throw std::runtime_error("RawFrameStore: pool smaller than frameCount");
			}
		}
	}
	catch (...) {
		// The destructor does not run for a throwing constructor: without this the
		// already pinned strips would stay locked until process exit.
		Release();
		throw;
	}
}

RawFrameStore::~RawFrameStore()
{
	pool_.stop();
	Release();
}

void RawFrameStore::Release() noexcept
{
	// Drop the headers first: a dangling RawFrame* then fails loudly instead of
	// reading recycled memory.
	for (RawFrame& f : frames_) {
		f.strip = nullptr;
		f.patches.clear();
	}
	for (void* p : pinned_)
		if (p) cudaFreeHost(p);
	pinned_.clear();
}

RingBuffer<RawFrame*>& RawFrameStore::Pool()
{
	return pool_;
}

std::size_t RawFrameStore::FrameCount() const
{
	return frames_.size();
}

std::size_t RawFrameStore::BytesPerFrame() const
{
	return layout_.PayloadBytes();
}

std::size_t RawFrameStore::TotalBytes() const
{
	return static_cast<size_t>(FrameCount() * BytesPerFrame());
}

const PatchLayout& RawFrameStore::Layout() const
{
	return layout_;
}
