#include "config.h"
#if USE_ENDTABLES
#define atomic_init(x,y) (x)->store(y)
#ifndef INFINITE
    constexpr DWORD INFINITE = 0xFFFFFFFF;
#endif
#include "tbprobe.c"
#endif /* USE_ENDTABLES */
