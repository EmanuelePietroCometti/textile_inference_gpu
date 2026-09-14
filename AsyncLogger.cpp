#include "AsyncLogger.h"
#include <cassert>
#include <cstdio>
#include <ctime>
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include "RealTimeConfig.h"

static bool s_ansiColorSupported = false;

namespace {
	constexpr std::string_view LevelName(LogLevel l)
	{
		switch (l) {
		case LogLevel::Warning: return "WARN";
		case LogLevel::Error:   return "ERROR";
		default:                return "INFO";
		}
	}
}

namespace { std::atomic<AsyncLogger*> g_instance{ nullptr }; }

namespace Log {
	namespace detail {
		void Dispatch(LogLevel level, fmt::string_view fmtStr, fmt::format_args args)
		{
			if (AsyncLogger* l = g_instance.load(std::memory_order_acquire)) {
				l->Write(level, fmtStr, args);
			}
		}
	}
}

static void EnableVTProcessing()
{
	s_ansiColorSupported = true;
	for (DWORD stdHandle : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
		HANDLE h = GetStdHandle(stdHandle);
		DWORD mode = 0;
		if (h == NULL || h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) {
			s_ansiColorSupported = false;
			continue;
		}
		if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
			s_ansiColorSupported = false;
		}
	}
}

AsyncLogger::AsyncLogger(size_t maxMessageChars, size_t queueCapacity, size_t flushIntervalMs, size_t notifyThreshold) : 
	maxMessageChars(maxMessageChars),
	queueCapacity(queueCapacity),
	flushIntervalMs(flushIntervalMs),
	notifyThreshold(notifyThreshold),
	queue(queueCapacity, LogRecord(maxMessageChars))
{
	AsyncLogger* expected = nullptr;
	assert(g_instance.compare_exchange_strong(expected, this) && "una sola istanza");
}

AsyncLogger::~AsyncLogger()
{
	AsyncLogger* self = this;
	g_instance.compare_exchange_strong(self, nullptr);
	Stop();
}

void AsyncLogger::Start()
{
	bool expected = false;
	if (!running.compare_exchange_strong(expected, true))
	{
		return;
	}
	EnableVTProcessing();
	consumer = std::thread(&AsyncLogger::ConsumerLoop, this);
}

void AsyncLogger::Stop()
{
	bool expected = true;
	if (!running.compare_exchange_strong(expected, false))
	{
		return;
	}
	queue.stop();
	if (consumer.joinable()) {
		consumer.join();
	}
	
	std::fflush(stdout);
	std::fflush(stderr);
}

std::string FormatLogLine(std::chrono::system_clock::time_point ts,
	LogLevel level, std::string_view msg)
{
	const auto tt = std::chrono::system_clock::to_time_t(ts);
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		ts.time_since_epoch()) % 1000;

	std::tm tm{};
	localtime_s(&tm, &tt);

	return fmt::format("[{:02}:{:02}:{:02}.{:03}] [{}] {}",
		tm.tm_hour, tm.tm_min, tm.tm_sec, ms.count(),
		LevelName(level), msg);
}

void AsyncLogger::WriteRecord(const LogRecord& record)
{
	std::string line = FormatLogLine(record.timestamp(), record.level(), record.message());
	if (record.truncated()) line += " [TRUNCATED]";

	switch (record.level())
	{
	case LogLevel::Warning:
		fmt::print(s_ansiColorSupported ? fg(fmt::color::yellow) : fmt::text_style{},
			"{}\n", line);
		break;
	case LogLevel::Error:
		fmt::print(stderr,
			s_ansiColorSupported ? fg(fmt::color::red) | fmt::emphasis::bold
			: fmt::text_style{},
			"{}\n", line);
		break;
	default:
		fmt::print("{}\n", line);
		break;
	}
}

void AsyncLogger::Write(LogLevel level, fmt::string_view fmtStr, fmt::format_args args)
{
	const auto ts = std::chrono::system_clock::now();

	thread_local LogRecord scratch(maxMessageChars);

	thread_local fmt::memory_buffer tmp;
	tmp.clear();
	fmt::vformat_to(fmt::appender(tmp), fmtStr, args);

	scratch.assign(level, ts, std::string_view(tmp.data(), tmp.size()));

	if (!queue.try_push_swap(scratch)) {
		dropped.fetch_add(1, std::memory_order_relaxed);
	}
}

void AsyncLogger::ConsumerLoop()
{
	const DWORD rtError = RT::ConfigureBackgroundThread();
	const bool isSuccess = (rtError == 0);

	LogRecord record(maxMessageChars);

	// Writes directly, bypassing the queue: the consumer thread is the only
	// one that can drain it, and it might have already been stopped at this point.
	const std::string line = isSuccess
		? "[RT] Consumer thread priority: THREAD_PRIORITY_LOWEST"
		: fmt::format("[RT] THREAD_PRIORITY_LOWEST not granted (error: {}).", rtError);

	const LogLevel level = isSuccess ? LogLevel::Info : LogLevel::Warning;

	record.assign(level, std::chrono::system_clock::now(), line);
	WriteRecord(record);
	while (queue.pop(record)) {
		WriteRecord(record);
	}
	std::fflush(stdout);
}

uint64_t AsyncLogger::DroppedCount()
{

	return  dropped.load(std::memory_order_relaxed);
}