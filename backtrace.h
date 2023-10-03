#pragma once

#if __APPLE__ || __linux__
#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>
#include <memory>

template <int MAX_FRAMES = 256>
void backtrace(std::ostream &out)
{
    void *frames[MAX_FRAMES] = {0};

    int num_frames = backtrace(frames, MAX_FRAMES);
    auto deleter = [](void *p)
    { if (p) free(p); };

    std::unique_ptr<char *[], decltype(deleter)>
        symbols(backtrace_symbols(frames, num_frames), deleter);

    if (!symbols)
        num_frames = 0;

    for (int i = 0; i != num_frames; ++i)
    {
        out << symbols[i] << std::endl;
    }
}
#else

template <int MAX_FRAMES = 256>
void backtrace(std::ostream &out)
{
    /* TODO: Windows */
}
#endif /* __APPLE__ || __linux__ */