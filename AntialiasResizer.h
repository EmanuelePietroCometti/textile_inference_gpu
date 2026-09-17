#pragma once
#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief High-performance image resizer utilizing 16-bit fixed-point arithmetic.
 *
 * @details This class is heavily optimized for real-time computer vision pipelines.
 * By converting standard floating-point interpolation weights into fixed-point integers
 * (Q14 format), it maximizes SIMD vectorization throughput and eliminates expensive
 * floating-point arithmetic from the critical path. It enforces a strict zero-allocation
 * policy by utilizing pre-allocated scratch and destination buffers.
 */
class AntialiasResizer {
public:
    /**
     * @brief Fixed-point scale shift (number of fractional bits).
     * @details Using 14 bits for the fractional part (Q14 format) ensures that coefficients
     * fit safely into `std::int16_t`, preventing overflow when multiplied with 8-bit pixels
     * and accumulated into a 32-bit integer register.
     */
    static constexpr int kWeightShift = 14;

    /**
     * @brief The integer representation of 1.0 in the current fixed-point scale.
     */
    static constexpr int kWeightOne = 1 << kWeightShift;

    /**
     * @brief Constructs the resizer and precomputes the fixed-point anti-aliasing coefficients.
     *
     * @param inW Expected width of the source image.
     * @param inH Expected height of the source image.
     * @param outW Desired width of the output image.
     * @param outH Desired height of the output image.
     */
    AntialiasResizer(int inW, int inH, int outW, int outH);

    /**
     * @brief Applies the precomputed fixed-point resizing operation.
     *
     * @details Executes a two-pass separable convolution (typically horizontal then vertical).
     * To maintain zero-allocation, it requires both a final destination matrix and a temporary
     * scratch matrix to hold the intermediate 16-bit or 32-bit values between passes.
     *
     * @param src The input image (must match `inW` and `inH`).
     * @param dst The pre-allocated output image.
     * @param scratch The pre-allocated intermediate buffer.
     */
    void Resize(const cv::Mat& src, cv::Mat& dst, cv::Mat& scratch) const;

    /**
     * @brief Validates if the given dimensions match the precomputed input dimensions.
     *
     * @param srcW Width of the incoming frame.
     * @param srcH Height of the incoming frame.
     * @return `true` if dimensions match, `false` otherwise.
     */
    bool Matches(int srcW, int srcH) const;

    /**
     * @brief Retrieves the preconfigured output width.
     * @return The width of the resized image.
     */
    int OutW() const;

    /**
     * @brief Retrieves the preconfigured output height.
     * @return The height of the resized image.
     */
    int OutH() const;

    /**
     * @brief Factory method to generate a correctly sized and typed scratch buffer.
     *
     * @details The scratch buffer must accommodate intermediate accumulation values
     * before the final bit-shift and clamping.
     *
     * @return A pre-allocated `cv::Mat` suitable for the `scratch` parameter in `Resize()`.
     */
    cv::Mat MakeScratch() const;

    /**
     * @brief Factory method to generate a correctly sized destination buffer.
     *
     * @return A pre-allocated `cv::Mat` suitable for the `dst` parameter in `Resize()`.
     */
    cv::Mat MakeDestination() const;

private:
    /**
     * @brief Calculates interpolation boundaries and fixed-point weights.
     *
     * @param inSize The size of the input dimension.
     * @param outSize The size of the output dimension.
     * @param bounds Output vector storing the starting source indices for each target pixel.
     * @param weights Output vector storing the Q14 fixed-point weights.
     * @param ksize Output parameter storing the kernel size.
     */
    static void PrecomputeCoeffs(int inSize, int outSize, std::vector<int>& bounds, std::vector<std::int16_t>& weights, int& ksize);

    /**
     * @brief Restores the fixed-point accumulator back to an 8-bit pixel space.
     *
     * @details Performs a right bit-shift (`>> kWeightShift`) to scale down the accumulated
     * integer value, then clamps it strictly to the [0, 255] range to prevent artifacting.
     *
     * @param acc The 32-bit integer accumulation of (pixel_value * fixed_weight).
     * @return The finalized 8-bit unsigned integer pixel.
     */
    static uint8_t RoundShiftClipU8(int acc);

    int inW_, inH_, outW_, outH_, hK_, vK_;

    std::vector<int> hBounds;                     ///< Starting X coordinates in the source image.
    std::vector<int> vBounds;                     ///< Starting Y coordinates in the source image.
    std::vector<std::int16_t> hWeights;           ///< Precomputed horizontal weights in Q14 fixed-point format.
    std::vector<std::int16_t> vWeights;           ///< Precomputed vertical weights in Q14 fixed-point format.

    bool identity_;                               ///< Flag indicating if dimensions match (enables direct copy).
};