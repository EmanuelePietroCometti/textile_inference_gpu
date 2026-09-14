#pragma once

#include <vector>
#include <chrono>
#include <cstdint>
#include <string_view>

#include <fmt/format.h>

/**
 * @brief Represents the severity level of a log message.
 */
enum class LogLevel : uint8_t {
    Info = 0,       ///< Standard informational messages.
    Warning = 1,    ///< Non-critical issues or unexpected behaviors.
    Error = 2       ///< Critical failures that may impact system execution.
};

/**
 * @brief Pre-allocated container for a single log event.
 *
 * @details Designed specifically for use within a Ring Buffer (Object Pool pattern).
 * By pre-allocating the internal text buffer upon construction, this class ensures
 * that the critical path (producer thread) performs zero heap allocations when recording logs.
 * Instead of creating new objects, existing instances are overwritten using the `assign` method.
 */
class LogRecord
{
public:
    /**
     * @brief Constructs a log record and pre-allocates its internal buffer.
     *
     * @param maxMessageChars The maximum number of characters this record can store.
     * Any message exceeding this limit will be truncated.
     */
    explicit LogRecord(std::size_t maxMessageChars);

    /**
     * @brief Overwrites the current record with new log data.
     *
     * @details This is the core method for the zero-allocation strategy. It copies the
     * contents of `msg` into the pre-allocated internal buffer. If `msg.size()` exceeds
     * `maxMessageChars`, the payload is safely truncated and the `truncated_` flag is set.
     *
     * @param level Severity level of the new log.
     * @param ts The exact timestamp when the log was generated.
     * @param msg The formatted message payload to copy into the buffer.
     */
    void assign(LogLevel level, std::chrono::system_clock::time_point ts, std::string_view msg);

    /**
     * @brief Retrieves a lightweight view of the currently stored message.
     *
     * @return A `std::string_view` pointing to the valid data within the internal buffer.
     */
    std::string_view message() const;

    /**
     * @brief Retrieves the severity level of this record.
     *
     * @return The current `LogLevel`.
     */
    LogLevel level() const;

    /**
     * @brief Retrieves the timestamp of this record.
     *
     * @return The system clock timestamp.
     */
    std::chrono::system_clock::time_point timestamp() const;

    /**
     * @brief Checks if the original message was truncated during the `assign` operation.
     *
     * @return `true` if truncation occurred, `false` otherwise.
     */
    bool truncated() const;

    /** @brief Capacity of the pre-allocated buffer, in characters. */
    std::size_t capacity() const;
private:
    std::vector<char> buffer_; ///< Pre-allocated memory buffer to store the message payload without heap allocations.
    std::chrono::system_clock::time_point timestamp_{}; ///< Timestamp of the logged event.
    std::uint32_t length_ = 0;                          ///< Actual length of the currently stored message.
    LogLevel level_ = LogLevel::Info;                   ///< Severity level of the current log.
    bool truncated_ = false;                            ///< Flag indicating if the last assigned message was truncated.
};