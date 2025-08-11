#pragma once

#if __APPLE__ || __linux__
#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>
#include <memory>

template <int MAX_FRAMES = 256>
void dump_backtrace(std::ostream &out)
{
#if !__ANDROID__ || __ANDROID_API__ >= 33
    void *frames[MAX_FRAMES] = {};

    int num_frames = backtrace(frames, MAX_FRAMES);
    auto deleter = [](void *p)
    { if (p) free(p); };

    std::unique_ptr<char *[], decltype(deleter)>
        symbols(backtrace_symbols(frames, num_frames), deleter);

    if (!symbols)
    {
        num_frames = 0;
        out << "backtrace_symbols failed" << std::endl;
    }
    for (int i = 0; i != num_frames; ++i)
    {
        out << symbols[i] << std::endl;
    }
#endif /* !__ANDROID__ || __ANDROID_API__ >= 33 */
}
#else

#include <string>
#include "ms_windows.h"
#pragma comment(lib, "user32.lib")

template <int MAX_FRAMES = 256>
void dump_backtrace(std::ostream& /* out */)
{
    const std::string msg = "Attach debugger to process: " + std::to_string(GetCurrentProcessId());

    MessageBoxA(HWND(0), msg.c_str(), "Sturddle Chess Engine Error", MB_OK | MB_ICONERROR);
    DebugBreak();
}
#endif /* __APPLE__ || __linux__ */
