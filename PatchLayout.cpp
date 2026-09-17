#include "PatchLayout.h"

#include <stdexcept>

PatchLayout::PatchLayout(const PatchGeometry& g) :
    layout_(g)
{
    // Check for zero fields to avoid zero-size allocations or underflows
    if (layout_.stripWidth == 0 || layout_.stripHeight == 0 ||
        layout_.patchHeight == 0 || layout_.count == 0)
    {
        throw std::invalid_argument("PatchGeometry dimensions and count must be strictly greater than zero.");
    }

    // Check channel validity (only Grayscale or RGB allowed)
    if (layout_.channels != 1 && layout_.channels != 3)
    {
        throw std::invalid_argument("PatchGeometry channels must be exactly 1 or 3.");
    }

    // Check overlap validity to prevent underflow in Step() calculation
    if (layout_.overlap >= layout_.patchHeight)
    {
        throw std::invalid_argument("PatchGeometry overlap must be strictly less than patchHeight.");
    }

    // Check geometry consistency
    // Safe to calculate Step() now because count > 0 and patchHeight > overlap
    if ((layout_.count - 1) * Step() + layout_.patchHeight != layout_.stripHeight)
    {
        throw std::invalid_argument("PatchGeometry is inconsistent: (count-1) * step + patchHeight does not match stripHeight.");
    }
}

std::uint32_t PatchLayout::Step() const
{
	return layout_.patchHeight - layout_.overlap;
}

std::size_t PatchLayout::RowBytes() const
{
	return static_cast<std::size_t>(layout_.stripWidth)*layout_.channels;
}

std::size_t PatchLayout::PatchBytes() const
{
	return static_cast<std::size_t>(layout_.patchHeight) * RowBytes();
}

std::size_t PatchLayout::PayloadBytes() const
{
	return static_cast<std::size_t>(layout_.stripHeight) * RowBytes();
}

std::size_t PatchLayout::Offset(std::uint32_t i) const
{
	return  static_cast<std::size_t>(i) * Step() * RowBytes();
}

std::uint32_t PatchLayout::StripWidth() const
{
	return layout_.stripWidth;
}

std::uint32_t PatchLayout::StripHeight() const
{
	return layout_.stripHeight;
}

std::uint32_t PatchLayout::Channels() const
{
	return layout_.channels;
}

std::uint32_t PatchLayout::PatchHeight() const
{
	return layout_.patchHeight;
}

std::uint32_t PatchLayout::Overlap() const
{
	return layout_.overlap;
}

std::uint32_t PatchLayout::Count() const
{
	return layout_.count;
}