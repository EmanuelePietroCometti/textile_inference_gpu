#include "PerformanceMetrics.h"
#include "AsyncLogger.h"

PerformanceMetrics::PerformanceMetrics(int windowDimension, int batchSize) :
	windowDimension(windowDimension),
	batchSize(batchSize),
	completedBatches(0),
	preprocessing(windowDimension),
	batchPrep(windowDimension),
	gpu(windowDimension),
	h2d(windowDimension),
	run(windowDimension),
	d2h(windowDimension),
	postprocessing(windowDimension)
{
}

PerformanceMetrics::~PerformanceMetrics()
{
}

void PerformanceMetrics::addPreprocessingTime(double t)
{
	preprocessing.add(t);
}

void PerformanceMetrics::addBatchPrepTime(double t)
{
	batchPrep.add(t);
}

void PerformanceMetrics::addGpuTime(double t)
{
	gpu.add(t);
}

void PerformanceMetrics::addH2DTime(double t)
{
	h2d.add(t);
}

void PerformanceMetrics::addRunTime(double t)
{
	run.add(t);
}

void PerformanceMetrics::addD2HTime(double t)
{
	d2h.add(t);
}

void PerformanceMetrics::addPostprocessingTime(double t)
{
	postprocessing.add(t);
	completedBatches.fetch_add(1, std::memory_order_relaxed);
}

void PerformanceMetrics::printRollingAverage(int window)
{
	const uint64_t n = completedBatches.load(std::memory_order_relaxed);
	if (window <= 0 || n % (uint64_t)window != 0) return;

	Log::Info("[MONITOR] Batch {}-{} | per-batch(ms) CPU:{:.2f} DMA:{:.2f} | "
		"GPUwall:{:.2f} = H2D:{:.3f}+Run:{:.3f}+D2H:{:.3f} | Out:{:.2f}",
		n - (uint64_t)window + 1, n,
		preprocessing.average(), batchPrep.average(), gpu.average(),
		h2d.average(), run.average(), d2h.average(), postprocessing.average());
	PerformanceMetrics::clear();
}

PerformanceMetrics::Snapshot PerformanceMetrics::snapshot() const
{
	Snapshot s;
	s.preprocessing = preprocessing.average();
	s.batchPrep = batchPrep.average();
	s.gpu = gpu.average();
	s.h2d = h2d.average();
	s.run = run.average();
	s.d2h = d2h.average();
	s.postprocessing = postprocessing.average();
	s.completedBatches = completedBatches.load(std::memory_order_relaxed);
	return s;
}

void PerformanceMetrics::clear()
{
	preprocessing.clear();
	batchPrep.clear();
	gpu.clear();
	h2d.clear();
	run.clear();
	d2h.clear();
	postprocessing.clear();
	completedBatches.store(0, std::memory_order_relaxed);
}