#pragma once

#include <cstddef>
#include <cstdint>
#include <string> 

/**
 * @brief Defines the spatial dimensions and tiling strategy for image patch extraction.
 *
 * @details This structure holds the raw configuration required to slice a large continuous
 * image (or a continuous line-scan camera strip) into smaller, potentially overlapping,
 * sub-regions (patches) suitable for neural network inference.
 */
struct PatchGeometry {
    std::uint32_t stripWidth = 0;   ///< Width of the parent image strip in pixels.
    std::uint32_t stripHeight = 0;  ///< Total height of the parent image strip in pixels.
    std::uint32_t channels = 1;     ///< Number of color channels (e.g., 1 for Grayscale, 3 for RGB).
    std::uint32_t patchHeight = 0;  ///< Height of a single extracted patch in pixels.
    std::uint32_t overlap = 0;      ///< Number of overlapping pixels between consecutive patches.
    std::uint32_t count = 0;        ///< Total number of patches to be extracted from the strip.
};

/**
 * @brief Memory layout calculator for zero-copy patch extraction.
 *
 * @details Wraps a `PatchGeometry` configuration to provide fast, pre-calculated byte-level
 * metrics and memory offsets. This ensures safe and efficient pointer arithmetic when
 * feeding specific regions of a large image buffer directly into ML models (like ONNX/TensorRT)
 * without having to allocate and copy memory for each individual patch.
 */
class PatchLayout {
public:
    /**
     * @brief Constructs the layout calculator based on the provided geometry configuration.
     *
     * @param g The geometry specification detailing strip dimensions and patch sizes.
     */
    explicit PatchLayout(const PatchGeometry& g);

    /**
     * @brief Calculates the vertical stepping distance between consecutive patches.
     *
     * @details This is typically `patchHeight - overlap`. It represents how many pixels
     * the sliding window moves forward to extract the next patch.
     *
     * @return The step size in pixels.
     */
    std::uint32_t Step() const;

    /**
     * @brief Calculates the memory size of a single row of pixels.
     *
     * @return The number of bytes per row (typically `stripWidth * channels`).
     */
    std::size_t RowBytes() const;

    /**
     * @brief Calculates the memory footprint of a single complete patch.
     *
     * @return The total number of bytes in one patch (`RowBytes() * patchHeight`).
     */
    std::size_t PatchBytes() const;

    /**
     * @brief Calculates the total memory required to hold the entire sequence of patches.
     *
     * @return The total payload size in bytes.
     */
    std::size_t PayloadBytes() const;

    /**
     * @brief Computes the exact memory address offset for a specific patch.
     *
     * @details Used for pointer arithmetic. By adding this offset to the base pointer
     * of the image strip, you get the direct pointer to the top-left pixel of the i-th patch.
     *
     * @param i The index of the patch (from 0 to `count - 1`).
     * @return The byte offset from the start of the memory buffer.
     */
    std::size_t Offset(std::uint32_t i) const;

    // --- Geometry Accessors ---

    /** @brief Retrieves the width of the parent strip. */
    std::uint32_t StripWidth() const;

    /** @brief Retrieves the total height of the parent strip. */
    std::uint32_t StripHeight() const;

    /** @brief Retrieves the number of color channels. */
    std::uint32_t Channels() const;

    /** @brief Retrieves the height of a single patch. */
    std::uint32_t PatchHeight() const;

    /** @brief Retrieves the pixel overlap between consecutive patches. */
    std::uint32_t Overlap() const;

    /** @brief Retrieves the total number of patches. */
    std::uint32_t Count() const;

private:
    PatchGeometry layout_; ///< The internal geometry configuration instance.
};