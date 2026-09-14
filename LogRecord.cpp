#include "LogRecord.h"
#include <cassert>


LogRecord::LogRecord(std::size_t maxMessageChars) : buffer_(maxMessageChars)
{
	assert(maxMessageChars > 0 && "zero record discard every message");
}

std::string_view LogRecord::message() const
{
	return {
		buffer_.data(),
		length_
	};
}

LogLevel LogRecord::level() const
{
	return level_;
}

std::chrono::system_clock::time_point LogRecord::timestamp() const
{
	return timestamp_;
}
bool LogRecord::truncated() const 
{
	return truncated_;
}

void LogRecord::assign(LogLevel level, std::chrono::system_clock::time_point ts, std::string_view msg)
{
	const std::size_t n = std::min(msg.size(), buffer_.size());
	if (n > 0) {
		std::memcpy(buffer_.data(), msg.data(), n);
	}

	timestamp_ = ts;
	level_ = level;
	length_ = static_cast<std::uint32_t>(n);
	truncated_ = msg.size() > buffer_.size();
}