#include "RollingAverage.h"
#include <stdexcept>

RollingField::RollingField(size_t windowSize) :
	window(windowSize)
{
	if (windowSize == 0)
		throw std::invalid_argument("RollingField: windowSize deve essere > 0");
}

void RollingField::add(double value)
{
	std::lock_guard<std::mutex> lock(mutex);
	window[writeIndex] = value;
	writeIndex = (writeIndex + 1) % window.size();
	if (count < window.size()) ++count;
}

double RollingField::average() const
{
	std::lock_guard<std::mutex> lock(mutex);
	if (count == 0) return 0.0;
	double sum = 0.0;
	for (std::size_t i = 0; i < count; i++) sum += window[i];
	return sum / static_cast<double>(count);
}

std::size_t RollingField::samples() const
{
	std::lock_guard<std::mutex> lock(mutex);
	return count;
}

void RollingField::clear()
{
	std::lock_guard<std::mutex> lock(mutex);
	count = 0;
	writeIndex = 0;
}