#include "PerformanceMetrics.h"
#include "AsyncLogger.h"

PerformanceMetrics::PerformanceMetrics(int windowDimension) :
	windowDimension(windowDimension),
	completedBatches(0),
	preprocessing(windowDimension),
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

std::int64_t PerformanceMetrics::addPostprocessingTime(double t)
{
	postprocessing.add(t);
	// fetch_add returns the previous value, so each caller owns a distinct index.
	return completedBatches.fetch_add(1, std::memory_order_relaxed) + 1;
}

void PerformanceMetrics::printRollingAverage(std::int64_t batchIndex) const
{
	if (windowDimension <= 0 || batchIndex % windowDimension != 0) return;

	Log::Info("[MONITOR] Batch {}-{} | per-batch(ms) CPU:{:.2f} | "
		"GPUwall:{:.2f} = H2D:{:.3f}+Run:{:.3f}+D2H:{:.3f} | Out:{:.2f}",
		batchIndex - windowDimension + 1, batchIndex,
		preprocessing.average(), gpu.average(),
		h2d.average(), run.average(), d2h.average(), postprocessing.average());
}

PerformanceMetrics::Snapshot PerformanceMetrics::snapshot() const
{
	Snapshot s;
	s.preprocessing = preprocessing.average();
	s.gpu = gpu.average();
	s.h2d = h2d.average();
	s.run = run.average();
	s.d2h = d2h.average();
	s.postprocessing = postprocessing.average();
	s.completedBatches = completedBatches.load(std::memory_order_relaxed);
	return s;
}

void PerformanceMetrics::logFinalReport(const FinalReportInput& in) const
{
	const Snapshot s = snapshot();
	const std::int64_t completed = s.completedBatches;
	const std::uint64_t offered = in.framesRead + in.framesDropped;
	const double dropPct = offered > 0
		? 100.0 * static_cast<double>(in.framesDropped) / static_cast<double>(offered) : 0.0;
	const double rate = in.listenedSec > 0.0
		? static_cast<double>(completed) / in.listenedSec : 0.0;
	const std::int64_t lostInPipeline = static_cast<std::int64_t>(in.framesRead) - completed;
	const std::int64_t window = (std::min)(static_cast<std::int64_t>(windowDimension), completed);

	// Theoretical ceiling of a stage: threads / mean service time per batch.
	const auto capacity = [](std::uint32_t threads, double ms) {
		return ms > 0.0 ? threads * 1000.0 / ms : 0.0;
		};
	struct Stage { const char* name; double ms; double cap; };
	const Stage stages[] = {
		{ "prep",  s.preprocessing,  capacity(in.prepThreads,  s.preprocessing) },
		{ "infer", s.gpu,            capacity(in.inferThreads, s.gpu) },
		{ "post",  s.postprocessing, capacity(in.postThreads,  s.postprocessing) },
	};
	const Stage* slowest = nullptr;
	for (const Stage& st : stages)
		if (st.cap > 0.0 && (!slowest || st.cap < slowest->cap)) slowest = &st;

	Log::Info("{:=<40}", "");
	Log::Info("{:^40}", "FINAL REPORT");
	Log::Info("{:=<40}", "");

	Log::Info("{:-<40}", "-- Throughput ");
	Log::Info("  {:<26}{:>12.1f}", "Listening time [s]", in.listenedSec);
	Log::Info("  {:<26}{:>12}", "Batches completed", completed);
	Log::Info("  {:<26}{:>12.2f}", "Throughput [batch/s]", rate);

	Log::Info("{:-<40}", "-- Frames ");
	Log::Info("  {:<26}{:>12}", "Read from MMF", in.framesRead);
	Log::Info("  {:<26}{:>12}", "Dropped (no raw buffer)", in.framesDropped);
	Log::Info("  {:<26}{:>12.2f}", "Drop rate [%]", dropPct);
	Log::Info("  {:<26}{:>12}", "Lost in pipeline", lostInPipeline);
	Log::Info("  {:<26}{:>12}", "Lost log messages", in.lostLogMessages);

	Log::Info("{:-<40}", fmt::format("-- Stage latency (mean of last {}) ", window));
	Log::Info("  {:<14}{:>10}{:>14}", "stage", "mean [ms]", "max [batch/s]");
	Log::Info("  {:<14}{:>10.2f}{:>14.1f}",
		fmt::format("prep x{}", in.prepThreads), stages[0].ms, stages[0].cap);
	Log::Info("  {:<14}{:>10.2f}{:>14.1f}",
		fmt::format("infer x{}", in.inferThreads), stages[1].ms, stages[1].cap);
	Log::Info("    {:<12}{:>10.3f}", "h2d", s.h2d);
	Log::Info("    {:<12}{:>10.3f}", "run", s.run);
	Log::Info("    {:<12}{:>10.3f}", "d2h", s.d2h);
	Log::Info("  {:<14}{:>10.2f}{:>14.1f}",
		fmt::format("post x{}", in.postThreads), stages[2].ms, stages[2].cap);
	Log::Info("  {:<26}{:>12}", "Slowest stage", slowest ? slowest->name : "n/a");
	Log::Info("{:=<40}", "");
}