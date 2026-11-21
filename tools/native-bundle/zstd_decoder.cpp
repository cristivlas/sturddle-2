/*
 * zstd decompressor as a single translation unit, after
 * zstd/build/single_file_libs/zstddeclib-in.c
 */
#define DEBUGLEVEL 0
#define MEM_MODULE
#undef  XXH_NAMESPACE
#define XXH_NAMESPACE ZSTD_
#undef  XXH_PRIVATE_API
#define XXH_PRIVATE_API
#undef  XXH_INLINE_ALL
#define XXH_INLINE_ALL
#define ZSTD_LEGACY_SUPPORT 0
#define ZSTD_TRACE 0
#define ZSTD_DISABLE_ASM 1

#define ZSTD_DEPS_NEED_MALLOC
#include "zstd/lib/common/zstd_deps.h"

#include "zstd/lib/common/debug.c"
#include "zstd/lib/common/entropy_common.c"
#include "zstd/lib/common/error_private.c"
#include "zstd/lib/common/fse_decompress.c"
#include "zstd/lib/common/zstd_common.c"

#include "zstd/lib/decompress/huf_decompress.c"
#include "zstd/lib/decompress/zstd_ddict.c"
#include "zstd/lib/decompress/zstd_decompress.c"
#include "zstd/lib/decompress/zstd_decompress_block.c"
