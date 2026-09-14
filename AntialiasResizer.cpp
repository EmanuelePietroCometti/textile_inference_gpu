#include "AntialiasResizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

AntialiasResizer::AntialiasResizer(int inW, int inH, int outW, int outH) :
	inW_(inW),
	inH_(inH),
	outW_(outW),
	outH_(outH),
	hK_(0),
	vK_(0),
	identity_(inW == inW_ && inH == inH_)
{
	PrecomputeCoeffs(inW, outW, hBounds, hWeights, hK_);
	PrecomputeCoeffs(inH, outH, vBounds, vWeights, vK_);
}

void AntialiasResizer::Resize(const cv::Mat& src, cv::Mat& dst, cv::Mat& scratch) const
{
	CV_Assert(src.type() == CV_8UC3);
	CV_Assert(Matches(src.cols, src.rows));

	const int inH = inH_, inW = inW_;
	const int outW = outW_, outH = outH_;

	// Identity fast path. With inSize == outSize the triangle filter collapses
	// to weights [1, 0] anchored on bounds[o] == o for every output pixel, so
	// the two passes below reduce to a pixel-for-pixel copy. Running them anyway
	// costs ~1.2M double MACs + ~400k std::floor per 256x256 image, which is the
	// single largest item in the CPU preprocessing budget. Aliasing instead of
	// copying keeps this at zero cost; dst is read-only for the caller.
	if (identity_) {
		dst = src;
		return;
	}

	// Horizontal pass: [inH x inW] -> [inH x outW]. Scratch is LOCAL, so this
	// function is safe to call concurrently from the batch parallel_for.
	CV_Assert(scratch.rows == inH_ && scratch.cols == outW_ && scratch.type() == CV_8UC3);
	cv::Mat & hpass = scratch;
	for (int y = 0; y < inH; ++y) {
		const uint8_t* srow = src.ptr<uint8_t>(y);
		uint8_t* hrow = hpass.ptr<uint8_t>(y);
		for (int o = 0; o < outW; ++o) {
			const int s = hBounds[o];
			const int avail = (std::min)(hK_, inW - s);
			const double* w = &hWeights[static_cast<size_t>(o) * hK_];
			double a0 = 0.0, a1 = 0.0, a2 = 0.0;
			for (int t = 0; t < avail; ++t) {
				const uint8_t* px = srow + static_cast<size_t>(s + t) * 3;
				const double wt = w[t];
				a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
			}
			uint8_t* op = hrow + static_cast<size_t>(o) * 3;
			op[0] = RoundClipU8(a0); 
			op[1] = RoundClipU8(a1); 
			op[2] = RoundClipU8(a2);
		}
	}

	// Vertical pass: [inH x outW] -> [outH x outW]. create() is a no-op when
	// dst already has the correct shape and type (see MakeDestination).
	dst.create(outH, outW, CV_8UC3);
	for (int o = 0; o < outH; ++o) {
		const int s = vBounds[o];
		const int avail = (std::min)(vK_, inH - s);
		const double* w = &vWeights[static_cast<size_t>(o) * vK_];
		uint8_t * drow = dst.ptr<uint8_t>(o);
		for (int x = 0; x < outW; ++x) {
			double a0 = 0.0, a1 = 0.0, a2 = 0.0;
			for (int t = 0; t < avail; ++t) {
				const uint8_t * px = hpass.ptr<uint8_t>(s + t) + static_cast<size_t>(x) * 3;
				const double wt = w[t];
				a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
			
			}
			uint8_t * op = drow + static_cast<size_t>(x) * 3;
			op[0] = RoundClipU8(a0);
			op[1] = RoundClipU8(a1);
			op[2] = RoundClipU8(a2);

		}
	}
}

uint8_t AntialiasResizer::RoundClipU8(double v)
{
	double r = std::floor(v + 0.5);
	if (r < 0.0) r = 0.0;
	if (r > 255.0) r = 255.0;
	return static_cast<uint8_t>(r);
}

bool AntialiasResizer::Matches(int srcW, int srcH) const
{
	return srcW == inW_ && srcH == inH_;
}

int AntialiasResizer::OutW() const
{
	return outW_;
}

int AntialiasResizer::OutH() const
{
	return outH_;
}

void AntialiasResizer::PrecomputeCoeffs(int inSize, int outSize, std::vector<int>& bounds, std::vector<double>& weights, int& ksize)
{
	const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
	const double filterscale = scale >= 1.0 ? scale : 1.0;
	const double support = 1.0 * filterscale;
	const double ss = 1.0 / filterscale;

	ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
	bounds.assign(outSize, 0);
	weights.assign(static_cast<size_t>(outSize) * ksize, 0.0);

	for (int o = 0; o < outSize; ++o)
	{
		const double center = (o + 0.5) * scale;
		int xmin = static_cast<int>(center - support + 0.5);
		if (xmin < 0) xmin = 0;
		int xmax = static_cast<int>(center + support + 0.5);
		if (xmax > inSize) xmax = inSize;
		
		const int n = xmax - xmin;
		double total = 0.0;
		for (int t = 0; t < n; ++t)
		{
			double w = 1.0 - std::abs((xmin + t - center + 0.5) * ss);
			if (w < 0.0) w = 0.0;
			weights[static_cast<size_t>(o) * ksize + t] = w;
			total += w;
		}

		if (total > 0.0)
			for (int t = 0; t < n; ++t) weights[static_cast<size_t>(o) * ksize + t] /= total;
		bounds[o] = xmin;
	}
}

cv::Mat AntialiasResizer::MakeScratch() const
{
	cv::Mat m(inH_, outW_, CV_8UC3);
	std::memset(m.data, 0, m.total() * m.elemSize());
	return m;
}

cv::Mat AntialiasResizer::MakeDestination() const
{
	if (identity_) return cv::Mat();   // il fast path aliasa src, il buffer sarebbe sprecato
	cv::Mat m(outH_, outW_, CV_8UC3);
	std::memset(m.data, 0, m.total() * m.elemSize());   // commit delle pagine
	return m;
}