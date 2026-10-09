#include "ConsoleKeys.h"

#include <cwctype>
#include <stdexcept>
#include <string>

ConsoleKeys::ConsoleKeys()
{
    h_ = GetStdHandle(STD_INPUT_HANDLE);
    if (h_ == nullptr || h_ == INVALID_HANDLE_VALUE || !GetConsoleMode(h_, &oldMode_))
        throw std::runtime_error("stdin is not an interactive console: keyboard control unavailable");

    const DWORD mode = (oldMode_ & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
        | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT | ENABLE_QUICK_EDIT_MODE))
        | ENABLE_EXTENDED_FLAGS;
    if (!SetConsoleMode(h_, mode))
        throw std::runtime_error("SetConsoleMode failed, GetLastError=" + std::to_string(GetLastError()));

    FlushConsoleInputBuffer(h_); // ignore keys typed during the (long) initialization
}

ConsoleKeys::~ConsoleKeys() { SetConsoleMode(h_, oldMode_); }

wchar_t ConsoleKeys::Poll(DWORD timeoutMs)
{
    if (WaitForSingleObject(h_, timeoutMs) != WAIT_OBJECT_0) return 0;

    // One record per call: if more are pending, the next Poll returns immediately,
    // so no key is lost when several events are queued at once.
    INPUT_RECORD rec{};
    DWORD n = 0;
    if (!ReadConsoleInputW(h_, &rec, 1, &n) || n == 0) return 0;
    if (rec.EventType != KEY_EVENT || !rec.Event.KeyEvent.bKeyDown) return 0;

    return static_cast<wchar_t>(std::towlower(rec.Event.KeyEvent.uChar.UnicodeChar));
}
