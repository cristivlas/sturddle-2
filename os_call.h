#pragma once

#include <string>
#include <string_view>
#include <system_error>

#if _WIN32
  #include "ms_windows.h"

  inline int last_os_error() { return static_cast<int>(GetLastError()); }

  struct FileHandle
  {
      static bool ok(HANDLE h) { return h != INVALID_HANDLE_VALUE; }
  };

  /* ResumeThread, WaitForSingleObject, GetFileSize */
  struct DwordResult
  {
      static bool ok(DWORD r) { return r != DWORD(-1); }
  };
#else
  #include <cerrno>
  #include <sys/mman.h>

  inline int last_os_error() { return errno; }

  struct PosixResult
  {
      template <typename T> static bool ok(T r) { return r != T(-1); }
  };

  struct MappedPtr
  {
      static bool ok(void* p) { return p != MAP_FAILED; }
  };
#endif /* _WIN32 */


/* BOOL, null HANDLE or pointer, zero DWORD */
struct NonZero
{
    template <typename T> static bool ok(T r) { return r != T(); }
};


[[noreturn]] inline void throw_last_error(std::string_view api)
{
    const int error = last_os_error();
    throw std::system_error(error, std::system_category(), std::string(api) + " failed");
}


template <typename T, typename Check>
class OsResult
{
public:
    OsResult(T value) : _value(value) {}

    explicit operator bool() const { return Check::ok(_value); }

    T value_or_throw(std::string_view api) const
    {
        if (!*this)
            throw_last_error(api);
        return _value;
    }

private:
    T _value;
};


#define OS_CALL_CHECK(check, fn, ...) \
    OsResult<decltype(::fn(__VA_ARGS__)), check>(::fn(__VA_ARGS__)).value_or_throw(#fn)

#define OS_CALL(fn, ...) OS_CALL_CHECK(NonZero, fn, __VA_ARGS__)
