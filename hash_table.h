#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>

#if _WIN32
  #include "ms_windows.h"
#else
  #include <sys/mman.h>
#endif /* !_WIN32 */

constexpr size_t HASH_TABLE_MAX_READERS = 64;
constexpr int SPIN_LOCK_MAX_RETRY = 1024 * 1024;

namespace
{
/* Allocators */
#if _MSC_VER
static constexpr auto CACHE_LINE_SIZE = std::hardware_destructive_interference_size;

#elif defined(__APPLE__) && defined(__aarch64__)
/* Apple M1 */
static constexpr std::size_t CACHE_LINE_SIZE = 128;
#else
static constexpr std::size_t CACHE_LINE_SIZE = 64;
#endif /* _MSC_VER */

template <typename T>
class cache_line_allocator
{
public:
    using value_type = T;
    using pointer = T *;

    cache_line_allocator() = default;

    template <typename U>
    struct rebind
    {
        using other = cache_line_allocator<U>;
    };

    template <typename U>
    cache_line_allocator(const cache_line_allocator<U> &) noexcept {}

    INLINE pointer allocate(std::size_t n)
    {
#if _WIN32
        auto p = _aligned_malloc(n * sizeof(T), CACHE_LINE_SIZE);
        if (!p)
            throw std::bad_alloc();
#else
        void *p = nullptr;
        if (posix_memalign(&p, CACHE_LINE_SIZE, n * sizeof(T)) != 0)
            throw std::bad_alloc();
#endif /* !_WIN32 */

        return reinterpret_cast<pointer>(p);
    }

    INLINE void deallocate(pointer p, std::size_t)
    {
#if _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif /* _WIN32 */
    }

    template <typename U, typename... Args>
    INLINE void construct(U *p, Args &&...args) { new (p) U(std::forward<Args>(args)...); }

    template <typename U>
    INLINE void destroy(U *p) { p->~U(); }
};

template <typename T, typename U>
INLINE bool operator==(const cache_line_allocator<T> &, const cache_line_allocator<U> &) noexcept
{
    return true;
}

template <typename T, typename U>
INLINE bool operator!=(const cache_line_allocator<T> &a, const cache_line_allocator<U> &b) noexcept
{
    return !(a == b);
}

template <typename T>
class mmap_allocator
{
public:
    static constexpr size_t PAGE_SIZE = 4096;

    using value_type = T;
    using pointer = T *;

    mmap_allocator() = default;

    template <typename U>
    struct rebind
    {
        using other = mmap_allocator<U>;
    };

    template <typename U>
    mmap_allocator(const mmap_allocator<U> &) noexcept {}

private:
#if _WIN32
    /* Stored before user data */
    struct alignas(CACHE_LINE_SIZE) allocation_header
    {
        size_t size;
        HANDLE mapping_handle;
        size_t user_size;
    };
#endif

public:
    INLINE pointer allocate(std::size_t n)
    {
#if _WIN32
        constexpr size_t header_size = sizeof(allocation_header);
        static_assert(header_size == CACHE_LINE_SIZE);
        const auto user_bytes = n * sizeof(value_type);
        const auto total_bytes = ((header_size + user_bytes + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

        HANDLE h_mapping = ::CreateFileMapping(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            static_cast<DWORD>(total_bytes >> 32),
            static_cast<DWORD>(total_bytes & 0xFFFFFFFF),
            nullptr);

        if (!h_mapping)
            throw std::bad_alloc();

        void* ptr = ::MapViewOfFile(h_mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_bytes);
        if (!ptr)
        {
            ::CloseHandle(h_mapping);
            throw std::bad_alloc();
        }

        // Store header info
        allocation_header* header = static_cast<allocation_header*>(ptr);
        header->size = header_size;
        header->mapping_handle = h_mapping;
        header->user_size = user_bytes;

        // Return aligned pointer after the header
        return reinterpret_cast<pointer>(header + 1);

#else
        const auto bytes = ((n * sizeof(value_type) + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        void* ptr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED)
            throw std::bad_alloc();

        return reinterpret_cast<pointer>(ptr);
#endif /* !_WIN32 */
    }

    INLINE void deallocate(pointer p, std::size_t n)
    {
#if _WIN32
        if (p)
        {
            // Get the header that precedes the user data
            allocation_header* header = reinterpret_cast<allocation_header*>(p) - 1;
            ASSERT_ALWAYS(header->size == sizeof(allocation_header));
            HANDLE h_mapping = header->mapping_handle;

            // Unmap the view starting from the header
            ::UnmapViewOfFile(header);
            ::CloseHandle(h_mapping);
        }
#else
        // POSIX implementation
        if (p)
        {
            const auto bytes = ((n * sizeof(value_type) + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
            ::munmap(p, bytes);
        }
#endif /* !_WIN32 */
    }

    template <typename U, typename... Args>
    INLINE void construct(U *p, Args &&...args) {
        new (p) U(std::forward<Args>(args)...);
    }

    template <typename U>
    INLINE void destroy(U *p) {
        p->~U();
    }
};

template <typename T, typename U>
INLINE bool operator==(const mmap_allocator<T> &, const mmap_allocator<U> &) noexcept
{
    return true;
}

template <typename T, typename U>
INLINE bool operator!=(const mmap_allocator<T> &a, const mmap_allocator<U> &b) noexcept
{
    return !(a == b);
}
} /* Allocators */

/*
 * Thomas Neumann's primes.hpp requires __int128
 * http://databasearchitects.blogspot.com/2020/01/all-hash-table-sizes-you-will-ever-need.html
 */
#if HAVE_INT128
#include "primes.hpp"

static INLINE size_t pick_prime(size_t n)
{
    return primes::Prime::pick(n).get();
}
#else
static INLINE size_t pick_prime(size_t n)
{
    return n;
}
#endif /* HAVE_INT128 */


INLINE uint64_t scramble64(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;

    return h;
}


namespace search
{
    template <typename T>
    class BaseLock
    {
    protected:
        std::atomic<T> *_mutex;
        bool _locked;

    public:
        BaseLock() : _mutex(nullptr), _locked(false) {}

        BaseLock(BaseLock &&other) : _mutex(other._mutex), _locked(other._locked)
        {
            other._locked = false;
        }
        explicit BaseLock(std::atomic<T> &mutex) : _mutex(&mutex), _locked(false)
        {
        }

        BaseLock &operator=(const BaseLock &) = delete;

        INLINE bool is_valid() const { return _locked; }
    };


    static constexpr auto LOCK_OK = std::memory_order_acq_rel;
    static constexpr auto LOCK_FAIL = std::memory_order_acquire;


    template <typename T, T Locked = std::numeric_limits<T>::max()>
    class UniqueLock : public BaseLock<T>
    {
    public:
        UniqueLock() = default;
        UniqueLock(UniqueLock &&other) = default;

        explicit UniqueLock(std::atomic<T> &mutex) : BaseLock<T>(mutex)
        {
#if SMP
            int i = 0;

            for (T unlocked = T();
                 !this->_mutex->compare_exchange_weak(unlocked, Locked, LOCK_OK, LOCK_FAIL);
                 unlocked = T())
            {
                if (++i > SPIN_LOCK_MAX_RETRY)
                    return;
            }
#endif /* SMP */

            this->_locked = true;
        }

        ~UniqueLock()
        {
#if SMP
            if (this->_locked)
            {
                this->_mutex->store(T(), std::memory_order_seq_cst);
            }
#endif /* SMP */
        }
    };

    template <typename T, T MaxShare, bool Blocking = true>
    class SharedLock : public BaseLock<T>
    {
    public:
        SharedLock() = default;
        SharedLock(SharedLock &&other) = default;

        explicit SharedLock(std::atomic<T> &mutex) : BaseLock<T>(mutex)
        {
#if SMP
            while (true)
            {
                for (T i = T(); i < MaxShare; )
                {
                    const auto j = i + 1;
                    if (this->_mutex->compare_exchange_weak(i, j, LOCK_OK, LOCK_FAIL))
                    {
                        this->_locked = true;
                        return;
                    }
                    if constexpr(Blocking)
                        ;
                    else if (i > MaxShare)
                        break;
                }
            }
#else
            this->_locked = true;
#endif /* SMP */
        }

        ~SharedLock()
        {
#if SMP
            if (this->_locked)
            {
                auto value = --(*this->_mutex);
                ASSERT(value >= T());
            }
#endif /* SMP */
        }
    };


    template <typename T, size_t BUCKET_SIZE = std::max<size_t>(128, CACHE_LINE_SIZE) / sizeof(T)>
    class hash_table
    {
        template <size_t SIZE>
        struct alignas(128) Bucket
        {
            using lock_state_t = int8_t;

            lock_state_t        _lock_state;
            uint8_t             _used = 0;
            uint32_t            _clock = 0;
            std::array<T, SIZE> _entries;

            std::atomic<lock_state_t> &mutex()
            {
                static_assert(sizeof(std::atomic<lock_state_t>) == sizeof(_lock_state));
                return *reinterpret_cast<std::atomic<lock_state_t> *>(&_lock_state);
            }

            static constexpr size_t size() { static_assert(SIZE); return SIZE; }
        };

        using bucket_t = Bucket<BUCKET_SIZE>;

    #if USE_MMAP_HASH_TABLE
        using allocator_type = mmap_allocator<bucket_t>;
    #else
        using allocator_type = cache_line_allocator<bucket_t>;
    #endif

        using data_t = std::vector<bucket_t, allocator_type>;

        using shared_lock_t = SharedLock<typename bucket_t::lock_state_t, HASH_TABLE_MAX_READERS>;
        using unique_lock_t = UniqueLock<typename bucket_t::lock_state_t>;

        uint32_t _clock = 0;
        count_t _used = 0;
        data_t _data; /* table entries */

    private:
        /* Allocation helper, find the closest prime number of buckets that fits in. */
        static INLINE size_t get_num_buckets(size_t megabytes, size_t mem_avail)
        {
            static_assert(sizeof(T) == 20);
            static_assert(bucket_t::size() == 6);
            static_assert(bucket_size() == 128);

            auto buckets = megabytes * ONE_MEGABYTE / bucket_size();
            auto prime_buckets = pick_prime(buckets);

            while (prime_buckets * bucket_size() > mem_avail)
            {
                if (buckets == 0)
                    return 0;

                prime_buckets = pick_prime(--buckets);
            }
            return prime_buckets;
        }

        INLINE bucket_t &get_bucket(uint64_t hash)
        {
            return _data[scramble64(hash) % _data.size()];
        }

    public:
        static constexpr size_t bucket_size() { return sizeof(bucket_t); }

        using entry_t = T;

        template <typename L>
        class Proxy
        {
            using lock_t = L;

            entry_t *_entry;
            lock_t _lock;

        public:
            Proxy() : _entry(nullptr) {}
            Proxy(entry_t *entry, lock_t &&lock) : _entry(entry), _lock(std::move(lock)) {}

            INLINE explicit operator bool() const { return _lock.is_valid(); }
            INLINE const entry_t *operator->() const
            {
                ASSERT(_entry);
                return _entry;
            }
            INLINE const entry_t &operator*() const
            {
                ASSERT(_entry);
                return *_entry;
            }
            INLINE entry_t *operator->()
            {
                ASSERT(_entry);
                return _entry;
            }
            INLINE entry_t &operator*()
            {
                ASSERT(_entry);
                return *_entry;
            }
        };

        hash_table(size_t megabytes, size_t mem_avail)
        {
            resize(megabytes, mem_avail);
        }

        hash_table(const hash_table &) = delete;
        hash_table &operator=(const hash_table &) = delete;

        INLINE size_t byte_capacity() const { return _data.size() * sizeof(_data[0]); }

        /* Capacity in entries of type T */
        INLINE size_t capacity() const { return _data.size() * bucket_t::size(); }

        void clear()
        {
            ++_clock; // O(1) -- buckets are lazily erased on next use
            _used = 0;
        }


        INLINE void increment_usage(bucket_t& bucket)
        {
#if SMP
            _used.fetch_add(1, std::memory_order_relaxed);
#else
            ++_used;
#endif
            ASSERT(bucket._used < bucket_t::size());
            ++bucket._used;
        }

        INLINE size_t size() const { return _used; }

        void resize(size_t megabytes, size_t mem_avail)
        {
            const auto buckets = get_num_buckets(megabytes, mem_avail + byte_capacity());

            if (buckets == 0)
                throw std::bad_alloc();

            _data.resize(buckets);
        }

        template <typename S, typename lock_t = shared_lock_t>
        INLINE Proxy<lock_t> lookup_read(const S &s)
        {
            const auto h = s.hash();
            ASSERT(h);

            auto &bucket = get_bucket(h);

            lock_t lock(bucket.mutex());
            if (lock.is_valid() && (bucket._clock == this->_clock))
            {
                for (auto &e : bucket._entries)
                {
                    if (e._hash == h)
                    {
                        return Proxy<lock_t>(&e, std::move(lock));
                    }
                }
            }
            return Proxy<lock_t>();
        }

        template <typename S, typename P, typename lock_t = unique_lock_t>
        INLINE Proxy<lock_t> lookup_write(const S &s, int depth, P&& priority)
        {
            const auto h = s.hash();
            ASSERT(h);

            auto &bucket = get_bucket(h);

            lock_t lock(bucket.mutex());
            if (lock.is_valid())
            {
                if (bucket._clock != this->_clock)
                {
                    bucket._entries.fill(entry_t());
                    bucket._clock = this->_clock;
                    bucket._used = 0;
                }

                entry_t *entry = &bucket._entries[0];
                auto lowest_priority = P::highest();

                for (auto &e : bucket._entries)
                {
                    if (!e.is_valid())
                    {
                        increment_usage(bucket);
                        entry = &e;
                        break;
                    }

                    const auto ep = e.priority();

                    if (e._hash == h)
                    {
                        if (ep > priority)
                        {
                            return Proxy<lock_t>();
                        }

                        entry = &e;
                        break;
                    }

                    if (ep < lowest_priority)
                    {
                        lowest_priority = ep;
                        entry = &e;
                    }
                }

                return Proxy<lock_t>(entry, std::move(lock));
            }

            return Proxy<lock_t>();
        }
    };
}
