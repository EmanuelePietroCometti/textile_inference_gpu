#pragma once

#include <Windows.h>

/**
 * @brief Raw, non-blocking keyboard reader on the console input buffer.
 *
 * @details Waits on the stdin handle with a timeout, so the same loop can also
 * drive the metrics cadence. Line buffering and echo are disabled (single key, no
 * Enter). QuickEdit is disabled too: with QuickEdit on, a mouse click in the console
 * freezes stdout, the logger thread blocks inside fmt::print and log records start
 * being dropped. The original console mode is restored by the destructor.
 */
class ConsoleKeys {
public:
    /// @throws std::runtime_error if stdin is not an interactive console.
    ConsoleKeys();

    ~ConsoleKeys();

    ConsoleKeys(const ConsoleKeys&) = delete;
    ConsoleKeys& operator=(const ConsoleKeys&) = delete;

    /// @return The lowercase character of the next key press, or 0 on timeout / non-key events.
    wchar_t Poll(DWORD timeoutMs);

private:
    HANDLE h_ = nullptr;
    DWORD oldMode_ = 0;
};
