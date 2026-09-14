#include "RealTimeConfig.h"
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#include "AsyncLogger.h"

RealTimeScope::RealTimeScope()
{
	previousClass = GetPriorityClass(GetCurrentProcess());
	if (previousClass == 0)
		Log::Warning("[RT] GetPriorityClass failed (error: {}); no restore on exit.", GetLastError());

	timerRaised = (timeBeginPeriod(1) == TIMERR_NOERROR);
	if (!timerRaised)
		Log::Warning("[RT] timeBeginPeriod(1) not granted; timer resolution unchanged.");

	const DWORD applied = RT::ApplyPriorityClass(REALTIME_PRIORITY_CLASS);
	if (applied == REALTIME_PRIORITY_CLASS)
	{
		realtime_ = elevated_ = true;
		Log::Info("[RT] Process priority class: REALTIME");
		return;
	}

	if (applied == HIGH_PRIORITY_CLASS)
	{
		elevated_ = true;
		Log::Warning("[RT] REALTIME requested but silently downgraded to HIGH "
			"(SeIncreaseBasePriorityPrivilege missing).");
		return;
	}

	if (applied == 0)
		Log::Warning("[RT] REALTIME class not granted (error: {}). Falling back to HIGH.", GetLastError());

	if (RT::ApplyPriorityClass(HIGH_PRIORITY_CLASS) == HIGH_PRIORITY_CLASS)
	{
		elevated_ = true;
		Log::Info("[RT] Process priority class: HIGH");
		return;
	}

	Log::Error("[RT] Neither REALTIME nor HIGH granted (error: {}). Running at default priority.", GetLastError());
}

RealTimeScope::~RealTimeScope()
{
	if (previousClass != 0 && GetPriorityClass(GetCurrentProcess()) != previousClass)
	{
		if (SetPriorityClass(GetCurrentProcess(), previousClass))
			Log::Info("[RT] Previous priority class restored.");
		else
			Log::Error("[RT] Restore of priority class failed (error: {}).", GetLastError());
	}

	if (timerRaised)
		timeEndPeriod(1);
}

bool RealTimeScope::elevated() const noexcept { return elevated_; }
bool RealTimeScope::realtime() const noexcept { return realtime_; }

DWORD RT::ConfigureBackgroundThread()
{
	// if frame drops are too much, raise the thread priority to BELOW_NORMAL
	if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST)) return 0;
	return GetLastError();
}

DWORD RT::ConfigureRealtimeThread()
{
	if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) return 0;
	return GetLastError();
}

DWORD RT::ApplyPriorityClass(DWORD requested)
{
	if (!SetPriorityClass(GetCurrentProcess(), requested)) return 0;
	return GetPriorityClass(GetCurrentProcess());
}