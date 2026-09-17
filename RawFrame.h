#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "PatchLayout.h"
#include "RingBuffer.h"

struct RawFrame {
	std::uint8_t* strip = nullptr;
	std::vector<cv::Mat> patches;
	std::uint64_t seq = 0;
	std::int64_t acquiredQPC = 0;
};


class RawFrameStore {
public:
	RawFrameStore(const PatchLayout& layout, std::size_t frameCount);
	~RawFrameStore();

	RawFrameStore(const RawFrameStore&) = delete;
	RawFrameStore& operator=(const RawFrameStore&) = delete;

	RingBuffer<RawFrame*>& Pool();
	std::size_t FrameCount() const;
	std::size_t BytesPerFrame() const;
	std::size_t TotalBytes() const;
	const PatchLayout& Layout() const;
private:
	PatchLayout layout_;
	std::vector<std::unique_ptr<std::uint8_t[]>> storage_;
	std::vector<RawFrame> frames_;
	RingBuffer<RawFrame*> pool_;
};