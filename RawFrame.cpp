#include "RawFrame.h"

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

	storage_.reserve(frameCount);
	frames_.resize(frameCount);

	for (std::size_t k = 0; k < frameCount; ++k)
	{
		storage_.push_back(std::make_unique_for_overwrite<std::uint8_t[]>(bytes));
		std::uint8_t* strip = storage_.back().get();
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

RawFrameStore::~RawFrameStore()
{
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
