#pragma once

#include <array>
#include <atomic>
#include <cstdlib>

#if _WIN32
  #include "ms_windows.h"
#else
  #include <sys/mman.h>
#endif /* !_WIN32 */
#include "utility.h"

constexpr size_t HASH_TABLE_MAX_READERS = 64;
constexpr int SPIN_LOCK_MAX_RETRY = 1024 * 1024;


namespace alloc
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


static INLINE size_t get_even(size_t n)
{
    return (n + 1) & ~1;
}


/* Extract upper 32 bits of hash for TT key (lower bits used for bucket index) */
INLINE uint32_t hash_key(uint64_t h)
{
    return static_cast<uint32_t>(h >> 32);
}


namespace search
{
    template <typename T>
    class BaseLock
    {
    protected:
        std::atomic<T> *_mutex = nullptr;
        bool _locked = false;

    public:
        BaseLock() = default;
        INLINE BaseLock(BaseLock &&other) : _mutex(other._mutex), _locked(other._locked)
        {
            other._locked = false;
        }
        explicit INLINE BaseLock(std::atomic<T> &mutex) : _mutex(&mutex), _locked(false)
        {
        }

        INLINE BaseLock &operator=(const BaseLock &) = delete;

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

        explicit INLINE UniqueLock(std::atomic<T> &mutex) : BaseLock<T>(mutex)
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

        INLINE ~UniqueLock()
        {
#if SMP
            if (this->_locked)
            {
                this->_mutex->store(T(), std::memory_order_seq_cst);
            }
#else
            this->_locked = false;
#endif /* SMP */
        }
    };

    template <typename T, T MaxShare, bool Blocking = true>
    class SharedLock : public BaseLock<T>
    {
    public:
        SharedLock() = default;
        SharedLock(SharedLock &&other) = default;

        explicit INLINE SharedLock(std::atomic<T> &mutex) : BaseLock<T>(mutex)
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

        INLINE ~SharedLock()
        {
#if SMP
            if (this->_locked)
            {
                auto value = --(*this->_mutex);
                ASSERT(value >= T());
            }
#else
            this->_locked = false;
#endif /* SMP */
        }
    };


    template <typename T, size_t BUCKET_SIZE = 18>
    class hash_table
    {
        using clock_t = uint8_t;
        static constexpr uint16_t MAGIC = 0xDA7A;

        template <size_t SIZE>
        struct alignas(64) Bucket
        {
            using lock_state_t = int8_t;

            uint16_t            _magic = MAGIC;
            lock_state_t        _lock_state = 0;
            clock_t             _clock = 0;
            std::array<T, SIZE> _entries;

            INLINE bool is_valid(clock_t clock) const
            {
                return _clock == clock && _magic == MAGIC;  // stale bucket or uninitialized?
            }

            INLINE std::atomic<lock_state_t> &mutex()
            {
                static_assert(sizeof(std::atomic<lock_state_t>) == sizeof(_lock_state));
                return *reinterpret_cast<std::atomic<lock_state_t> *>(&_lock_state);
            }

            static constexpr size_t size() { static_assert(SIZE); return SIZE; }
        };

        using bucket_t = Bucket<BUCKET_SIZE>;

    #if USE_MMAP_HASH_TABLE
        using allocator_type = alloc::mmap_allocator<bucket_t>;
    #else
        using allocator_type = alloc::cache_line_allocator<bucket_t>;
    #endif

        using data_t = std::vector<bucket_t, allocator_type>;

        using shared_lock_t = SharedLock<typename bucket_t::lock_state_t, HASH_TABLE_MAX_READERS>;
        using unique_lock_t = UniqueLock<typename bucket_t::lock_state_t>;

        clock_t _clock = 0;
        count_t _used = 0;
        uint8_t _generation = 0;
        data_t _data; /* table entries */

    private:
        static INLINE size_t get_num_buckets(size_t megabytes)
        {
            static_assert(sizeof(T) == 14);
            static_assert(bucket_size() == 256);

            auto buckets = megabytes * ONE_MEGABYTE / bucket_size();
            return get_even(buckets);
        }

        INLINE bucket_t &get_bucket(uint64_t hash)
        {
            ASSERT(!_data.empty());
            ASSERT(_data.size() % 2 == 0);

            const auto idx = hash & (_data.size() - 1);
            ASSERT(idx >= 0 && idx < _data.size());

            PREFETCH(&_data[idx]);
            return _data[idx];
        }

    public:
        static constexpr size_t bucket_size() { return sizeof(bucket_t); }

        using entry_t = T;

        explicit hash_table(size_t megabytes)
        {
            resize(megabytes);
        }

        hash_table(const hash_table &) = delete;
        hash_table &operator=(const hash_table &) = delete;

        INLINE size_t byte_capacity() const { return _data.size() * sizeof(_data[0]); }

        /* Capacity in entries of type T */
        INLINE size_t capacity() const { return _data.size() * bucket_t::size(); }

        void clear(bool wipe)
        {
            // Fully erase if reasonably sized
            if (wipe && _data.size() * sizeof(decltype(_data[0])) <= 1024 * ONE_MEGABYTE)
                std::fill_n(&_data[0], _data.size(), bucket_t());

            ++_clock; // O(1) -- buckets are lazily erased on write
            _used = 0;

            _generation = 0;
        }

        INLINE uint8_t generation() const { return _generation; }
        INLINE void increment_generation() { _generation = (_generation + 1) & 31; }

        INLINE void increment_usage()
        {
#if SMP
            _used.fetch_add(1, std::memory_order_relaxed);
#else
            ++_used;
#endif
        }

        INLINE size_t size() const { return _used; }

        void resize(size_t megabytes)
        {
            // Convert size in megabytes to size in buckets
            const auto buckets = get_num_buckets(megabytes);

            if (buckets == 0)
                throw std::bad_alloc();

            _data.resize(buckets, bucket_t());
        }

        struct Result
        {
            entry_t     _entry;
            int16_t     _replacement_slot = -1;
            bucket_t*   _bucket = nullptr;
        };


        template <typename S>
        INLINE Result probe(const S &s, int depth)
        {
            Result result;

            const auto h = s.hash();
            ASSERT(h);

            const auto key = hash_key(h);

            auto& bucket = get_bucket(h);
            shared_lock_t lock(bucket.mutex());

            if (lock.is_valid())
            {
                result._bucket = &bucket;

                if (!bucket.is_valid(_clock))
                {
                    result._replacement_slot = 0;
                }
                else
                {
                    int replacement_score = depth + 1;

                    for (size_t slot = 0; slot < bucket_t::size(); ++slot)
                    {
                        auto& e = bucket._entries[slot];

                        if (e._key == key)
                        {
                            result._entry = e;
                            result._replacement_slot = slot;
                            break;
                        }

                        if (!e.is_valid())
                        {
                            result._replacement_slot = slot;
                            break;
                        }

                        const int age = (32 + generation() - e._generation) & 31;
                        const int score = e._depth - 2 * age + e.is_exact();

                        if (replacement_score > score)
                        {
                            replacement_score = score;
                            result._replacement_slot = slot;
                        }
                    }
                }
            }

            return result;
        }

        INLINE void store(const Result& r)
        {
            ASSERT(r._replacement_slot >= 0 && r._entry.is_valid());
            ASSERT(r._bucket);
            // validate that bucket is in data range
            ASSERT(r._bucket >= &_data[0]);
            ASSERT(std::distance(&_data[0], r._bucket) >= 0);
            ASSERT(std::distance(&_data[0], r._bucket) < int(_data.size()));

            auto& bucket = *r._bucket;

            unique_lock_t lock(bucket.mutex());
            if (lock.is_valid())
            {
                if (!bucket.is_valid(_clock))
                {
                    // Lazily erase / initialize stale buckets
                    bucket._entries.fill(entry_t());
                    bucket._magic = MAGIC;
                    bucket._clock = _clock;
                }

                auto& e = bucket._entries[r._replacement_slot];

                if (!e.is_valid())
                {
                    increment_usage();
                }

                e = r._entry;
            }
        }
    };
}
