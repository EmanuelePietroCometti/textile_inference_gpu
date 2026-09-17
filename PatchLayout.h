#pragma once


#include <cstddef>
#include <cstdint>
#include <string> 

struct PatchGeometry {
	std::uint32_t stripWidth = 0;
	std::uint32_t stripHeight = 0;
	std::uint32_t channels = 1;
	std::uint32_t patchHeight = 0;
	std::uint32_t overlap = 0;
	std::uint32_t count = 0;
};


class PatchLayout {
public:
	explicit PatchLayout(const PatchGeometry& g);

	std::uint32_t Step() const;
	std::size_t RowBytes() const;
	std::size_t PatchBytes() const;
	std::size_t PayloadBytes() const;
	std::size_t Offset(std::uint32_t i) const;

	std::uint32_t StripWidth() const;
	std::uint32_t StripHeight() const;
	std::uint32_t Channels() const;
	std::uint32_t PatchHeight() const;
	std::uint32_t Overlap() const;
	std::uint32_t Count() const;
private:
	PatchGeometry layout_;
};