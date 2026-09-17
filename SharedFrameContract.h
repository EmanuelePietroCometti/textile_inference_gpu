#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <cstdint>

/**
 * @file SharedFrameContract.h
 * @brief Mirror of the memory layout defined by the producer.
 */
namespace sfc {

    inline constexpr UINT32 NUM_FRAMES = 16;
    inline constexpr UINT32 NUM_CHANNELS = 1;
    inline constexpr UINT32 PAYLOAD = 512 * 512 * NUM_CHANNELS;
    inline constexpr UINT32 FULL_PAYLOAD = PAYLOAD * NUM_FRAMES;

    inline constexpr wchar_t kMappingName[] = L"Local\\GrabBuffer_0";
    inline constexpr wchar_t kReadyEventName[] = L"Local\\GrabBuffer_0_Ready";

    /**
     * canRead: FALSE = free, the producer can write to it. The initial state is
     * established by zeroing the memory section, not by a constructor.
     *
     *   producer  InterlockedCompareExchange(&canRead, TRUE, FALSE) == FALSE
     *             -> copies data to payload_, then calls SetEvent(ready)
     *   consumer  WaitForSingleObject(ready) -> copies data from payload_
     *             -> InterlockedExchange(&canRead, FALSE)
     */
    struct Frame {
        volatile LONG canRead;
        uint8_t payload_[FULL_PAYLOAD];
    };

} // namespace sfc