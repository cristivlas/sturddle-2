/*
 * Sturddle Chess Engine (C) 2022, 2023, 2024 Cristian Vlasceanu
 * --------------------------------------------------------------------------
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * --------------------------------------------------------------------------
 * Third-party files included in this project are subject to copyright
 * and licensed as stated in their respective header notes.
 * --------------------------------------------------------------------------
 */
#pragma once

#include <cstdlib>
#include <new>

#define FULL_SIZE_LOCK false /* half-size => 32 bit, full => 64 bit */

#if _MSC_VER
    static constexpr auto CACHE_LINE_SIZE = std::hardware_destructive_interference_size;

#elif defined(__APPLE__) && defined(__aarch64__)
    /* Apple M1 */
    static constexpr size_t CACHE_LINE_SIZE = 128;
#else
    static constexpr size_t CACHE_LINE_SIZE = 64;
#endif /* _MSC_VER */


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



template <typename T>
class CacheLineAlignedAllocator
{
public:
    using value_type = T;
    using pointer = T*;

    CacheLineAlignedAllocator() = default;

    template <typename U>
    struct rebind { using other = CacheLineAlignedAllocator<U>; };

    template <typename U>
    CacheLineAlignedAllocator(const CacheLineAlignedAllocator<U>&) noexcept {}

    INLINE pointer allocate(std::size_t n)
    {
    #if _WIN32
        auto p = _aligned_malloc(n * sizeof(T), CACHE_LINE_SIZE);
        if(!p) throw std::bad_alloc();
    #else
        void* p = nullptr;
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
    INLINE void construct(U* p, Args&&... args) { new(p) U(std::forward<Args>(args)...); }

    template <typename U> INLINE void destroy(U* p) { p->~U(); }
};

template <typename T, typename U>
INLINE bool operator==(const CacheLineAlignedAllocator<T>&, const CacheLineAlignedAllocator<U>&) noexcept
{
    return true;
}

template <typename T, typename U>
INLINE bool operator!=(const CacheLineAlignedAllocator<T>& a, const CacheLineAlignedAllocator<U>& b) noexcept
{
    return !(a == b);
}


namespace search
{
#if FULL_SIZE_LOCK
    using raw_lock_t = uint64_t;

    static INLINE constexpr raw_lock_t key(uint64_t hash)
    {
        return hash;
    }
#else
    /* 32-bit lock: smaller TT_Entry at the expense of an extra match() */
    using raw_lock_t = uint32_t;

    static INLINE constexpr raw_lock_t key(uint64_t key)
    {
        return key & 0xFFFFFFFF;
    }
#endif /* FULL_SIZE_LOCK */

    static_assert(std::atomic<raw_lock_t>::is_always_lock_free);
    static constexpr raw_lock_t LOCKED = raw_lock_t(-1);


    template<typename T> class SharedHashTable
    {
        using entry_t = T;
        using data_t = std::vector<uint8_t, CacheLineAlignedAllocator<uint8_t>>;
        using lock_t = std::atomic<raw_lock_t>;

        static constexpr auto bucket_size = 5 * CACHE_LINE_SIZE;
        static constexpr auto entries_per_bucket = bucket_size / sizeof(entry_t);

    public:
        class SpinLock
        {
            entry_t* _entry = nullptr;
            bool _locked = false;

            INLINE void take_ownership()
            {
            #if !NO_ASSERT
                if (_locked)
                    entry()->_owner = this;
            #endif /* NO_ASSERT */
            }

    #if SMP
            static INLINE lock_t* lock_p(entry_t* e)
            {
                static_assert(sizeof(lock_t) == sizeof(raw_lock_t));
                return reinterpret_cast<lock_t*>(&e->_lock);
            }

            static INLINE bool try_read_lock(lock_t* lock, uint64_t hash)
            {
                ASSERT(lock);
                auto k = key(hash);
                return lock->compare_exchange_strong(
                        k,
                        LOCKED,
                        std::memory_order_acq_rel,  /* stronger order on success */
                        std::memory_order_acquire);
            }

            static INLINE bool try_write_lock(lock_t* lock, uint64_t hash)
            {
                ASSERT(lock);
                auto k = key(hash);
                /* Expected to be called within a retry loop.
                 * Use weak compare exchange, as the loop provides
                 * resiliency to spurious failures.
                 */
                return lock->compare_exchange_weak(
                    k,
                    LOCKED,
                    std::memory_order_release, /* on success */
                    std::memory_order_acquire); /* on failure */
            }

            static INLINE bool write_lock(entry_t* entry)
            {
                auto lock = lock_p(entry);
                for (auto h = entry->_hash; !try_write_lock(lock, h); h = entry->_hash)
                    ; 

                return true;
            }

            static INLINE bool read_lock(entry_t* entry, uint64_t hash)
            {
                return try_read_lock(lock_p(entry), hash);
            }

            INLINE void release()
            {
                ASSERT(_locked); /* caller must check! */
                const auto e = entry();
                ASSERT(*lock_p(e) == LOCKED);
                ASSERT(entry()->_owner == this);

                lock_p(e)->store(key(e->_hash), std::memory_order_seq_cst);
                _locked = false;
            }
    #else
            INLINE static bool write_lock(const entry_t*) { return true; }
            INLINE static bool read_lock(const entry_t* e, uint64_t key) { return (e->_hash == key); }
            INLINE void release() { _locked = false; }
    #endif /* !SMP */

        protected:
            SpinLock() = default;

            explicit SpinLock(entry_t* e) : _entry(e), _locked(write_lock(e))
            {
                take_ownership();
            }

            SpinLock(entry_t* e, uint64_t hash) : _entry(e), _locked(read_lock(e, hash))
            {
                take_ownership();
            }

            ~SpinLock()
            {
                if (_locked)
                    release();
            }

            SpinLock(SpinLock&& other)
                : _entry(other._entry)
                , _locked(other._locked)
            {
                other._locked = false; /* this takes ownership */
                take_ownership();
            }

            SpinLock(const SpinLock&) = delete;
            SpinLock& operator=(SpinLock&&) = delete;
            SpinLock& operator=(const SpinLock&) = delete;

            entry_t* entry() const { return _entry; }
            bool is_locked() const { return _locked; }

        public:
            explicit operator bool() const { return is_locked(); }
        };


        class Proxy : public SpinLock
        {
        public:
            Proxy() = default;

            Proxy(entry_t* e) : SpinLock(e) {}
            Proxy(entry_t* e, uint64_t hash) : SpinLock(e, hash) {}

            INLINE const entry_t* operator->() const { return SpinLock::entry(); }
            INLINE const entry_t& operator *() const { return *SpinLock::entry(); }

            INLINE entry_t& operator *() { return *SpinLock::entry(); }
        };


        /* Allocation helper */
        static INLINE size_t get_buckets(size_t megabytes, size_t mem_avail)
        {
            auto buckets = megabytes * 1024 * 1024 / bucket_size;
            auto prime_buckets = pick_prime(buckets);
            while (prime_buckets * bucket_size > mem_avail)
            {
                if (buckets == 0)
                    return 0;
                prime_buckets = pick_prime(--buckets);
            }
            return prime_buckets;
        }

    public:
        SharedHashTable(size_t megabytes, size_t mem_avail)
            : _buckets(get_buckets(megabytes, mem_avail))
        {
            if (_buckets == 0)
                throw std::bad_alloc();

            _data.resize(_buckets * bucket_size);
        }

        SharedHashTable(const SharedHashTable&) = delete;
        SharedHashTable& operator=(const SharedHashTable&) = delete;

        void clear()
        {
            if (_used > 0)
            {
                std::fill_n(&_data[0], _data.size(), data_t::value_type());
                _used = 0;
            }
        }

        bool resize(size_t megabytes, size_t mem_avail)
        {
            auto buckets = get_buckets(megabytes, mem_avail + byte_capacity());
            if (buckets == 0)
                return false;
            _data.resize(buckets * bucket_size);
            _buckets = buckets;
            return true;
        }

        INLINE Proxy lookup_read(const chess::State& state)
        {
            const auto h = state.hash();
            ASSERT(h);
            auto* const first = get_entry(h);
            auto* const last = first + entries_per_bucket;

            for (auto e = first; e < last; ++e)
            {
                ASSERT((const uint8_t*)(e + 1) <= &_data.back() + 1);
                Proxy p(e, h);

                /* using full-size lock? */
                if constexpr(sizeof(raw_lock_t) == sizeof(h))
                {
                    if (p)
                    {
                        ASSERT(p->matches(state));
                        return p;
                    }
                }
                else if (p && p->matches(state))
                {
                    return p;
                }
            }
            return Proxy();
        }

        INLINE Proxy lookup_write(const chess::State& state, int depth)
        {
            const auto h = state.hash();
            ASSERT(h);
            auto* entry = get_entry(h);
            auto* const last = entry + entries_per_bucket;

            for (auto e = entry; e < last; ++e)
            {
                ASSERT((const uint8_t*)(e + 1) <= &_data.back());

                Proxy q(e, h); /* try non-blocking locking 1st */
                if (q)
                    return q;

                Proxy p(e);
                if (!p)
                    continue;

                if (!p->is_valid())
                {
                    /* slot is unoccupied, bump up usage count */
                #if SMP
                    _used.fetch_add(1, std::memory_order_relaxed);
                #else
                    ++_used;
                #endif /* SMP */

                    return p;
                }

                if (p->_hash == h)
                    return p;

                if (p->_age != _clock)
                    return p;

                if (depth >= p->_depth)
                {
                    entry = e;
                    depth = p->_depth;
                }
            }
            return Proxy(entry);
        }

        INLINE size_t byte_capacity() const { return _data.size(); }
        INLINE size_t capacity() const { return _buckets * entries_per_bucket; }

        INLINE uint8_t clock() const { return _clock; }
        INLINE void increment_clock() { ++_clock; }

        INLINE size_t size() const { return _used; }

    private:
        INLINE entry_t* get_entry(uint64_t hash)
        {
            auto index = (hash % _buckets) * bucket_size;
            ASSERT(index + sizeof(entry_t) * entries_per_bucket <= _data.size());
            return reinterpret_cast<entry_t*>(&_data[index]);
        }

    private:
        uint8_t     _clock = 0;
        count_t     _used = 0;
        data_t      _data;
        size_t      _buckets;
    };
}
