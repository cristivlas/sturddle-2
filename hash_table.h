#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>

constexpr size_t HASH_TABLE_MAX_READERS = 64;
constexpr int SPIN_LOCK_MAX_RETRY = 1024 * 1024;

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
        template <size_t S>
        struct Bucket
        {
            using lock_state_t = int8_t;

            lock_state_t _lock_state;
            std::array<T, S> _entries;

            std::atomic<lock_state_t> &mutex()
            {
                static_assert(sizeof(std::atomic<lock_state_t>) == sizeof(_lock_state));
                return *reinterpret_cast<std::atomic<lock_state_t> *>(&_lock_state);
            }

            static constexpr size_t size() { static_assert(S); return S; }
        };

        using bucket_t = Bucket<BUCKET_SIZE>;
        using data_t = std::vector<bucket_t, cache_line_allocator<bucket_t>>;

        using shared_lock_t = SharedLock<typename bucket_t::lock_state_t, HASH_TABLE_MAX_READERS>;
        using unique_lock_t = UniqueLock<typename bucket_t::lock_state_t>;

        uint8_t _clock = 0;
        count_t _used = 0;
        data_t _data; /* table entries */

    private:
        static INLINE size_t get_num_buckets(size_t megabytes, size_t mem_avail)
        {
            auto buckets = megabytes * 1024 * 1024 / bucket_size();
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
            if (_used)
                std::fill_n(&_data[0], _data.size(), bucket_t());
            _used = 0;
        }

        INLINE uint8_t clock() const { return _clock; }
        INLINE void increment_clock() { ++_clock; }

        INLINE void increment_usage()
        {
#if SMP
            _used.fetch_add(1, std::memory_order_relaxed);
#else
            ++_used;
#endif
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
            if (lock.is_valid())
            {
                for (auto &e : bucket._entries)
                {
                    if (e._hash == h)
                        return Proxy<lock_t>(&e, std::move(lock));
                }
            }
            return Proxy<lock_t>();
        }

        template <typename S, typename lock_t = unique_lock_t>
        INLINE Proxy<lock_t> lookup_write(const S &s, int depth)
        {
            const auto h = s.hash();
            ASSERT(h);

            auto &bucket = get_bucket(h);

            lock_t lock(bucket.mutex());
            if (lock.is_valid())
            {
                entry_t *entry = &bucket._entries[0];

                for (auto &e : bucket._entries)
                {
                    if (!e.is_valid())
                    {
                        increment_usage();
                        entry = &e;
                        break;
                    }

                    if (e._hash == h || e._age != _clock)
                    {
                        entry = &e;
                        break;
                    }

                    if (depth >= e._depth)
                    {
                        entry = &e;
                        depth = e._depth;
                    }
                }

                return Proxy<lock_t>(entry, std::move(lock));
            }

            return Proxy<lock_t>();
        }
    };
}
