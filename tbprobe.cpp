#include "config.h"
#if USE_ENDTABLES
#define atomic_init(x,y) (x)->store(y)
#if _WIN32 && !defined(INFINITE)
    constexpr DWORD INFINITE = 0xFFFFFFFF;
#endif
#include "tbprobe.c"
#endif /* USE_ENDTABLES */
