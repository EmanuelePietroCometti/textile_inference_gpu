#pragma once
#include <opencv2/opencv.hpp>


class AntialiasResizer {
public:
	AntialiasResizer(int inW, int inH, int outW, int outH);


	void Resize(const cv::Mat& src, cv::Mat& dst, cv::Mat& scratch) const;
	bool Matches(int srcW, int srcH) const;
	int OutW() const;
	int OutH() const;
	cv::Mat MakeScratch() const;
private:
	static void PrecomputeCoeffs(int inSize, int outSize, std::vector<int>& bounds, std::vector<double>& weights, int& ksize);
	static uint8_t RoundClipU8(double v);
	int inW_, inH_, outW_, outH_, hK_, vK_;
	std::vector<int> hBounds, vBounds;
	std::vector<double> hWeights, vWeights;
	bool identity;
};