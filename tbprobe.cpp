#include "config.h"
#if USE_ENDTABLES
#define atomic_init(x,y) (x)->store(y)
#include "tbprobe.c"
#endif /* USE_ENDTABLES */
