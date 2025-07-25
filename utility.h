/*
 * Sturddle Chess Engine (C) 2022 - 2025 Cristian Vlasceanu
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

#include <random>
#include <stdexcept>
#include "Python.h"


enum class CancelReason
{
    PY_ERROR =  1,
    PY_SIGNAL = 2,
};

extern void cancel_search(CancelReason);

namespace
{
    /*
     * Utility for calling into Cython.
     * Not strictly needed if the cython functions are marked 'with gil'.
     */
    struct cython_wrapper
    {
        class GIL_State
        {
            PyGILState_STATE state;

        public:
            INLINE GIL_State() : state(PyGILState_Ensure())
            {
            }

            INLINE ~GIL_State()
            {
                if (PyErr_CheckSignals() != 0)
                {
                    cancel_search(CancelReason::PY_SIGNAL);
                }
                if (PyErr_Occurred())
                {
                    cancel_search(CancelReason::PY_ERROR);
                }
                PyGILState_Release(state);
            }
        };

        template <typename R, typename... Params, typename... Args>
        static INLINE R call(R (*fn)(Params...), Args&&... args)
        {
            GIL_State gil_state;
            ASSERT(fn);
            return fn(std::forward<Args>(args)...);
        }

        template <typename R, typename... Params, typename... Args>
        static INLINE R call_nogil(R (*fn)(Params...), Args&&... args) noexcept
        {
            try
            {
                return fn(std::forward<Args>(args)...);
            }
            catch (const std::exception& e)
            {
                GIL_State gil_state;
                PyErr_SetString(PyExc_RuntimeError, e.what());
            }
            catch (...)
            {
                GIL_State gil_state;
                PyErr_SetString(PyExc_RuntimeError, "C++ exception");
            }
            return R();
        }
    };


    /*
     * For sorting small vectors of small objects.
     */
    template<typename Iterator, typename Compare>
    INLINE void insertion_sort(Iterator first, Iterator last, Compare comp)
    {
        using std::swap;

        for (auto i = first; i != last; ++i)
        {
            for (auto j = i; j != first; --j)
            {
                if (comp(*j, *(j-1)))
                    swap(*j, *(j-1));
                else
                    break;
            }
        }
    }


    template<typename  I> INLINE void shift_left_2(I first, I last)
    {
        ASSERT(std::distance(first, last) >= 2);

        using V = typename std::iterator_traits<I>::value_type;

    #if __cplusplus >= 202002L
        /* C++20 */
        auto i = std::shift_left(first, last, 2);
    #else
        auto i = std::rotate(first, first + 2, last);
    #endif
        *i++ = V(); *i = V();
    }


    static INLINE int random_int(int low, int high)
    {
        static thread_local std::mt19937 gen(std::random_device{}());
        return std::uniform_int_distribution<int>(low, high)(gen);
    }


    template<typename F>
    class on_scope_exit
    {
    public:
        explicit on_scope_exit(F f) : _f(f) {}

        on_scope_exit(const on_scope_exit&) = delete;
        on_scope_exit& operator=(const on_scope_exit&) = delete;

        ~on_scope_exit() noexcept(false) { _f(); }

    private:
        F _f;
    };


    template<typename T, typename H = std::size_t> struct Hasher
    {
        using hash_type = H;

        constexpr hash_type operator()(const T& key) const
        {
            return key.hash();
        }
    };

    namespace profile
    {
        struct Overhead
        {
            std::chrono::high_resolution_clock::duration value;
            Overhead()
            {
                const auto start = std::chrono::high_resolution_clock::now();
                const auto end = std::chrono::high_resolution_clock::now();
                value = end - start;
            }
        };
        const static Overhead overhead;
    }

    template <typename T, int PRINT_INTERVAL = 100000>
    struct ProfileScope
    {
        static std::chrono::high_resolution_clock::duration total_time;
        static int num_calls;

        std::chrono::time_point<std::chrono::high_resolution_clock> _start;

        INLINE void tally()
        {
            const auto end = std::chrono::high_resolution_clock::now();
            total_time += (end - _start - profile::overhead.value);
            if (num_calls % PRINT_INTERVAL == 0)
            {
                const auto avg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total_time).count() / num_calls;
                const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count();
                std::clog << &num_calls << " calls: " << num_calls << ", total: " << total_ms << " ms" << ", avg: " << avg_ns << " ns" << std::endl;
            }
        }

        ProfileScope() : _start(std::chrono::high_resolution_clock::now()) { ++num_calls; }
        ~ProfileScope() { tally(); }
    };

    template <typename T, int PRINT_INTERVAL>
    std::chrono::high_resolution_clock::duration ProfileScope<T, PRINT_INTERVAL>::total_time{};

    template <typename T, int PRINT_INTERVAL>
    int ProfileScope<T, PRINT_INTERVAL>::num_calls = 0;


    template<typename T>
    struct StorageView
    {
        static_assert(std::is_trivially_destructible<T>::value);

        template <size_t N>
        static T& get(unsigned char (&buf)[N], bool& valid)
        {
            static_assert(sizeof(T) <= N);
            static_assert(alignof(T) <= alignof(std::max_align_t));

            if (!valid)
            {
                new (buf) T();
                valid = true;
            }
            return *reinterpret_cast<T*>(&buf[0]);
        }

        template<size_t N>
        static void store(unsigned char (&buf)[N], bool& valid, const T& value)
        {
            static_assert(sizeof(T) <= N);
            static_assert(alignof(T) <= alignof(std::max_align_t));

            new (buf) T(value);
            valid = true;
        }
    };
} /* namespace */
