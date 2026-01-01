/*
 * Sturddle Chess Engine (C) 2022 - 2026 Cristian Vlasceanu
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

#include <algorithm>
#include <limits>
#include <memory>
#include <thread>
#include "chess.h"
#include "hash_table.h"
#include "utility.h"


INLINE constexpr int16_t checkmated(int ply)
{
    return -CHECKMATE + ply;
}


INLINE constexpr int16_t checkmating(int ply)
{
    return CHECKMATE + ply;
}


INLINE constexpr bool is_valid(score_t score)
{
    return score > SCORE_MIN;
}


namespace search
{
    struct Context;
    class TranspositionTable;

    /*
     * Search algorithms
     */
    score_t negamax(Context&, TranspositionTable&);
    score_t mtdf(Context&, score_t, TranspositionTable&);
    score_t iterative(Context&, TranspositionTable&, int);

    INLINE score_t _negamax(Context& ctxt, TranspositionTable& tbl) noexcept
    {
        return cython_wrapper::call_nogil(negamax, ctxt, tbl);
    }
    INLINE score_t _mtdf(Context& ctxt, score_t score, TranspositionTable& tbl) noexcept
    {
        return cython_wrapper::call_nogil(mtdf, ctxt, score, tbl);
    }
    INLINE score_t _iterative(Context& ctxt, TranspositionTable& tbl, int max_iter_count) noexcept
    {
        return cython_wrapper::call_nogil(iterative, ctxt, tbl, max_iter_count);
    }

    using BaseMove = chess::BaseMove;
    using Color = chess::Color;
    using Move = chess::Move;
    using MovesList = chess::MovesList;
    using State = chess::State;

    /*
     * Butterfly table for history counts and counter-move heuristics.
     * https://www.chessprogramming.org/index.php?title=Butterfly_Boards
     */
    template<typename T> struct MoveTable
    {
        MoveTable()
        {
            clear();
        }

        INLINE void clear()
        {
            std::fill_n(&_table[0][0], 64 * 64, T());
        }

        INLINE T& operator[](const Move& move)
        {
            ASSERT(move);
            return _table[move.from_square()][move.to_square()];
        }


        INLINE const T& lookup(const Move& move) const
        {
            ASSERT(move);
            return _table[move.from_square()][move.to_square()];
        }

        T _table[64][64] = {};
    };


    template<typename T> struct PieceMoveTable
    {
        PieceMoveTable()
        {
            clear();
        }

        INLINE void clear()
        {
            std::fill_n(&_table[0][0], 7 * 64, T());
        }

        INLINE T& lookup(chess::PieceType piece_type, const Move& move)
        {
            return lookup_impl(piece_type, move);
        }

        INLINE const T& lookup(chess::PieceType piece_type, const Move& move) const
        {
            return const_cast<PieceMoveTable*>(this)->lookup_impl(piece_type, move);
        }

    private:
        INLINE T& lookup_impl(chess::PieceType piece_type, const Move& move)
        {
            ASSERT(piece_type != chess::PieceType::NONE);
            ASSERT(move);

            if (piece_type == chess::PieceType::NONE || move.is_none())
                return _table[0][0];

            return _table[piece_type][move.to_square()];
        }

        T _table[7][64] = {};
    };

    using KillerMoves = std::array<Move, 2>;
    using KillerMovesTable = std::array<KillerMoves, PLY_MAX>;


    enum class TT_Type : uint8_t
    {
        NONE = 0,
        UPPER,
        EXACT,
        LOWER,
    };


#pragma pack(push, 2)

    class TT_Entry
    {
    public:
        uint64_t    _hash = 0;
        TT_Type     _type : 2;
        uint8_t     _generation : 5;
        bool        _pv : 1;
        int8_t      _depth = std::numeric_limits<int8_t>::min();
        BaseMove    _best_move;
        BaseMove    _hash_move;
        int16_t     _eval = SCORE_MIN; /* static eval */
        int16_t     _value = SCORE_MIN; /* search score */

        TT_Entry() : _type(TT_Type::NONE), _generation(0), _pv(false) {}

        INLINE bool is_exact() const { return _type == TT_Type::EXACT; }
        INLINE bool is_lower() const { return _type == TT_Type::LOWER; }
        INLINE bool is_upper() const { return _type == TT_Type::UPPER; }
        INLINE bool is_valid() const { return _type != TT_Type::NONE; }

        INLINE bool matches(const State& state) const
        {
            return _hash == state.hash();
        }

        template<typename C>
        INLINE const int16_t* lookup_score(C& ctxt) const
        {
            if (_depth >= ctxt.depth())
            {
                ASSERT(is_valid());

                if (is_lower())
                {
                    ctxt._alpha = std::max<score_t>(ctxt._alpha, _value);
                }
                else if (is_upper())
                {
                    ctxt._beta = std::min<score_t>(ctxt._beta, _value);
                }
                else
                {
                    ctxt._alpha = ctxt._beta = _value;
                }

                if (ctxt._alpha >= ctxt._beta)
                {
                    if (is_upper())
                    {
                        return &ctxt._beta; // return the tightened beta
                    }
                    else
                    {
                        return &_value; // return TT value for lower/exact
                    }
                }
            }

            return nullptr;
        }
    };
#pragma pack(pop)


    using HashTable = hash_table<TT_Entry>;

    /*
     * Hash table, counter moves, historical counts.
     * Under SMP the hash table is shared between threads.
     */
    class TranspositionTable
    {
    #if USE_BUTTERFLY_TABLES
        using HistoryCounters = MoveTable<std::pair<int, int>>;
        using IndexedMoves = MoveTable<BaseMove>;
    #else
        using HistoryCounters = PieceMoveTable<std::pair<int, int>>;
        using IndexedMoves = PieceMoveTable<BaseMove>;
    #endif /* USE_BUTTERFLY_TABLES */

        using PlyHistoryCounters = std::array<MoveTable<std::pair<int, int>>, 2>;
        using PlyHistory = std::array<PlyHistoryCounters, PLY_HISTORY_MAX>;

        /* https://www.chessprogramming.org/Countermove_Heuristic */
        IndexedMoves        _countermoves[2];

        KillerMovesTable    _killer_moves; /* killer moves at each ply */
        HistoryCounters     _hcounters[2]; /* History heuristic counters. */
        static HashTable    _table;        /* shared hashtable */

        void clear(); /* clear search stats, bump up generation */

    public:
        TranspositionTable() = default;
        ~TranspositionTable() = default;

        int _tid = 0;
        int16_t _iteration = 0;
        int16_t _eval_depth = 0;
        PlyHistory _ply_history;

        /* search window bounds */
        score_t _w_alpha = SCORE_MIN;
        score_t _w_beta = SCORE_MAX;
        bool _reset_window = false;
        bool _analysis = false;

        /* Stats for current thread */
        size_t _check_nodes = 0;
        size_t _eval_count = 0;
        size_t _endgame_nodes = 0;
        size_t _futility_prune_count = 0;
        size_t _history_counters = 0;
        size_t _history_counters_hit = 0;
        size_t _hits = 0;
        size_t _killers = 0;
        size_t _late_move_prune_count = 0;
        size_t _nodes = 0;
        size_t _nps = 0; /* nodes per second */
        size_t _null_move_cutoffs = 0;
        size_t _null_move_failed = 0;
        size_t _null_move_not_ok = 0;
        size_t _reductions = 0;
        size_t _retry_reductions = 0;
        size_t _tb_hits = 0;

        unsigned _pass = 0; /* MTD(f) pass */

        void clear_history();

        /* Re-initialize before new search or new game*/
        void init(bool new_game);

        template<typename C>
        BaseMove lookup_countermove(const C& ctxt) const;

        const KillerMoves* get_killer_moves(int ply) const
        {
            ASSERT(ply < PLY_MAX);
            return &_killer_moves[ply];
        }

        template<typename C> const int16_t* lookup(C& ctxt);

        template<TT_Type=TT_Type::NONE, typename C=struct Context>
        void store(C& ctxt, int depth);

        void update_entry(Context&, TT_Entry&, TT_Type, int depth);
        bool update_and_store(Context&, TT_Type);

        template<typename C> void store_countermove(C& ctxt);
        void store_killer_move(const Context&);

        const std::pair<int, int>& historical_counters(const State&, Color, const Move&) const;
        float history_score(int ply, const State&, Color, const Move&) const;

        size_t hits() const { return _hits; }
        size_t nodes() const { return _nodes; }

        /* percent usage (size over capacity) */
        static double usage();

        /* nodes per second */
        size_t nps() const { return _nps; }
        void set_nps(size_t nps) { _nps = nps; }

        void history_update_cutoffs(const Move&);
        void history_update_non_cutoffs(const Move&);

        void update_stats(const Context&);

        static size_t max_hash_size();

        /* return size of hash table */
        static size_t get_hash_size();

        /* set hash table size in MB */
        static void set_hash_size(size_t);

        size_t tb_hits() const { return _tb_hits; }
    };


    /*
     * https://www.chessprogramming.org/History_Heuristic
     * https://www.chessprogramming.org/Relative_History_Heuristic
     */
    INLINE const std::pair<int, int>&
    TranspositionTable::historical_counters(
        const State& state,
        Color turn,
        const Move& move) const
    {
        ASSERT(move);

    #if USE_BUTTERFLY_TABLES
        return _hcounters[turn].lookup(move);
    #else
        const auto pt = state.piece_type_at(move.from_square());
        return _hcounters[turn].lookup(pt, move);
    #endif /* USE_BUTTERFLY_TABLES */
    }


    INLINE float
    TranspositionTable::history_score(
        int ply,
        const State& state,
        Color turn,
        const Move& move) const
    {
        float score = 0;
        if (ply < PLY_HISTORY_MAX)
        {
            const auto& h = _ply_history[ply][turn].lookup(move);
            if (h.second)
                score = h.first / double(h.second * HISTORY_SCORE_DIV);
        }
        const auto& counters = historical_counters(state, turn, move);
        ASSERT(counters.first <= counters.second);

        /* blend scores */
        return score + (counters.second < 1 ? 0 : (double(HISTORY_SCORE_MUL) * counters.first) / counters.second);
    }


    INLINE void TranspositionTable::history_update_non_cutoffs(const Move& move)
    {
        if (move)
        {
            ASSERT(move._state);
            ASSERT(!move._state->is_capture());

            const auto turn = !move._state->turn; /* side that moved */
        #if USE_BUTTERFLY_TABLES
            auto& counters = _hcounters[turn][move];
        #else
            const auto pt = move._state->piece_type_at(move.to_square());
            auto& counters = _hcounters[turn].lookup(pt, move);
        #endif /* USE_BUTTERFLY_TABLES */

            ++counters.second;
        }
    }


    INLINE void TranspositionTable::history_update_cutoffs(const Move& move)
    {
        ASSERT(move);
        ASSERT(move._state);
        ASSERT(!move._state->is_capture());

        const auto turn = !move._state->turn; /* side that moved */

    #if USE_BUTTERFLY_TABLES
        auto& counts = _hcounters[turn][move];
    #else
        const auto pt = move._state->piece_type_at(move.to_square());
        ASSERT(pt != chess::PieceType::NONE);
        auto& counts = _hcounters[turn].lookup(pt, move);
    #endif /* USE_BUTTERFLY_TABLES */
        ++counts.first;
        ++counts.second;
    }


    template<typename C>
    INLINE BaseMove TranspositionTable::lookup_countermove(const C& ctxt) const
    {
    #if USE_BUTTERFLY_TABLES
        return _countermoves[ctxt.turn()].lookup(ctxt._move);
    #else
        const auto pt = ctxt.state().piece_type_at(ctxt._move.to_square());
        return _countermoves[ctxt.turn()].lookup(pt, ctxt._move);
    #endif /* USE_BUTTERFLY_TABLES */
    }


    template<typename C>
    INLINE const int16_t* TranspositionTable::lookup(C& ctxt)
    {
        if (ctxt.is_root() || ctxt._excluded)
            return nullptr;

        /* expect repetitions to be dealt with before calling into this function */
        ASSERT(!ctxt.is_repeated());

        if (!ctxt.tt_entry().is_valid() || ctxt.tt_entry()._depth < ctxt.depth())
            StorageView<HashTable::Result>::store(ctxt._state->tt_result, ctxt._state->has_tt_result, _table.probe(ctxt.state(), ctxt.depth()));
        else
            ASSERT(ctxt.tt_entry()._hash == ctxt.state().hash());

        if (ctxt.tt_entry().is_valid())
        {
            ASSERT(ctxt.tt_entry().matches(ctxt.state()));

            if constexpr(EXTRA_STATS)
                ++_hits;
        }

        /* http://www.talkchess.com/forum3/viewtopic.php?topic_view=threads&p=305236&t=30788 */
        if (!ctxt.is_pv_node() && !ctxt.is_retry())
        {
            if (auto value = ctxt.tt_entry().lookup_score(ctxt))
            {
                ctxt._score = *value;
                return value;
            }
        }

        if (ctxt._move)
            ctxt.set_counter_move(lookup_countermove(ctxt));

        return nullptr;
    }


    template<TT_Type T, typename C>
    INLINE void TranspositionTable::store(C& ctxt, int depth)
    {
        ASSERT(ctxt._score > SCORE_MIN);
        ASSERT(ctxt._score < SCORE_MAX);

        if (ctxt.tt_result()._replacement_slot >= 0)
        {
            auto type = T;

            /* type unknown at compile-time? */
            if constexpr(T == TT_Type::NONE)
            {
                type = TT_Type::EXACT;
                if (ctxt._score >= ctxt._beta)
                {
                    type = TT_Type::LOWER;
                }
                else if (ctxt._score <= ctxt._alpha)
                {
                    type = TT_Type::UPPER;
                }
            }

            update_entry(ctxt, ctxt.tt_entry(), type, depth);
            _table.store(ctxt.tt_result());
        }
    }


    template<typename C>
    INLINE void TranspositionTable::store_countermove(C& ctxt)
    {
        if (ctxt._move)
        {
            ASSERT(ctxt._cutoff_move);
    #if USE_BUTTERFLY_TABLES
            _countermoves[ctxt.turn()][ctxt._move] = ctxt._cutoff_move;
    #else
            const auto pt = ctxt.state().piece_type_at(ctxt._move.to_square());
            _countermoves[ctxt.turn()].lookup(pt, ctxt._move) = ctxt._cutoff_move;
    #endif /* USE_BUTTERFLY_TABLES */

            ctxt.set_counter_move(ctxt._cutoff_move);
        }
    }


    void stop_threads();

} /* namespace search */
