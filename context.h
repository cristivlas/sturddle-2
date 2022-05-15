/*
 * Sturddle Chess Engine (C) 2022 Cristi Vlasceanu
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
 *--------------------------------------------------------------------------
 */
#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set> /* unordered_multiset */
#include "Python.h"
#include "config.h"
#include "intrusive.h"
#include "search.h"
#include "utility.h"

/* Configuration API */
struct Param { int val = 0; int min_val; int max_val; std::string group; };

extern std::map<std::string, Param> _get_param_info();
extern void _set_param(const std::string&, int value, bool echo=false);
extern std::map<std::string, int> _get_params();


namespace search
{
    using time = std::chrono::time_point<std::chrono::steady_clock>;

    using Bitboard = chess::Bitboard;
    using Color = chess::Color;
    using Square = chess::Square;

    using ContextPtr = intrusive_ptr<struct Context>;
    using HistoryPtr = intrusive_ptr<struct History>;


    enum Algorithm : int
    {
        NEGAMAX,
        NEGASCOUT,
        MTDF,
    };


    /* For detecting repeated positions */
    struct History : public RefCounted<History>
    {
        History() = default;

        void insert(const State& s) { _positions.insert(s); }
        size_t count(const State& s) const { return _positions.count(s); }

        std::unordered_multiset<State, Hasher<State>> _positions;
        int _fifty = 0;
    };


    /*
     * Helper for making and ordering moves
     */
    class MoveMaker
    {
    public:
        static constexpr size_t MAX_MOVE = 2 * PLY_MAX;

        /* Note: top-half of the _moves buffers is reserved for do_exchanges */
        static THREAD_LOCAL MovesList _moves[MAX_MOVE];

        MoveMaker() = default;

        const Move* get_next_move(Context& ctxt, score_t futility = 0);

        int current(Context&);

        bool has_moves(Context&);
        bool have_skipped_moves() { return _have_pruned_moves || _have_quiet_moves; }
        bool is_last(Context&);
        bool is_singleton(Context&);

        void set_ply(int ply) { _ply = ply; }

        int rewind(Context&, int where, bool reorder);

        /* SMP: copy the root moves from the main thread to the other workers. */
        void set_initial_moves(const MovesList& moves);

        const MovesList& get_moves() const
        {
            ASSERT(_count >= 0);
            return moves();
        }

    private:
        void ensure_moves(Context&);

        void generate_unordered_moves(Context&);
        const Move* get_move_at(Context& ctxt, int index, score_t futility = 0);

        void make_capture(Context&, Move&);
        bool make_move(Context&, Move&, score_t futility = 0);
        bool make_move(Context&, Move&, MoveOrder, score_t = 0);
        void mark_as_illegal(Move&);
        void order_moves(Context&, size_t start_at, score_t futility);
        void sort_moves(Context&, size_t start_at);

        inline MovesList& moves() const
        {
            ASSERT(size_t(_ply) < MAX_MOVE);
            return _moves[_ply];
        }

        inline std::vector<State>& states() const
        {
            ASSERT(size_t(_ply) < MAX_MOVE);
            return _states[_ply];
        }

        MoveMaker(const MoveMaker&) = delete;
        MoveMaker& operator=(const MoveMaker&) = delete;

        int         _ply = 0;
        int         _count = -1;
        int         _current = -1;
        int         _phase = 0; /* move ordering phase */
        bool        _group_quiet_moves = false;
        bool        _have_move = false;
        bool        _have_quiet_moves = false;
        bool        _have_pruned_moves = false;
        bool        _need_sort = false;
        size_t      _state_index = 0;
        MovesList   _initial_moves;

        static THREAD_LOCAL std::vector<State> _states[MAX_MOVE];
    };


    inline void MoveMaker::mark_as_illegal(Move& move)
    {
        move._group = MoveOrder::ILLEGAL_MOVES;

        ASSERT(_count > 0);
        --_count;
    }


    inline void MoveMaker::sort_moves(Context& /* ctxt */, size_t start_at)
    {
        ASSERT(start_at < moves().size());

        auto& moves_list = moves();
#if 0
        /*
         * Walk backwards skipping over quiet, pruned, and illegal moves.
         */
        auto n = moves_list.size();
        for (; n > start_at; --n)
        {
            if (moves_list[n-1]._group < MoveOrder::QUIET_MOVES)
                break;
        }
        ASSERT(n == moves_list.size() || moves_list[n]._group >= MoveOrder::QUIET_MOVES);
        _count = n;
        const auto last = moves_list.begin() + n;
#else
        const auto last = moves_list.end();
#endif
        const auto first = moves_list.begin() + start_at;

        insertion_sort(first, last, [&](const Move& lhs, const Move& rhs)
            {
                return (lhs._group == rhs._group && lhs._score > rhs._score)
                    || (lhs._group < rhs._group);
            });

        _need_sort = false;
    }


    enum class LMRAction : int
    {
        None = 0,
        Ok,
        Prune,
    };

    /* Reason for retrying */
    enum class RETRY : uint8_t
    {
        None = 0,
        Reduced,
        PVS,
    };


    /*
     * The context of a searched node.
     */
    struct Context : public RefCounted<Context>
    {
        friend class RefCounted<Context>;

    private:
        ~Context();

    public:
        Context() = default;
        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        static void* operator new(size_t);
        static void* operator new(size_t, void*);
        static void operator delete(void*, size_t) noexcept;

        ContextPtr clone(int ply = 0) const;

        /* "parent" move in the graph; is "opponent" a better name? */
        Context*    _parent = nullptr;
        int         _tid = 0;
        int         _ply = 0;
        int         _max_depth = 0;

        Algorithm   _algorithm = Algorithm::NEGAMAX;

        score_t     _alpha = SCORE_MIN;
        score_t     _beta = SCORE_MAX;
        score_t     _score = SCORE_MIN; /* dynamic eval score */
        score_t     _retry_beta = SCORE_MAX; /* NEGASCOUT only */

        bool        _futility_pruning = true;
        bool        _is_null_move = false; /* for null-move pruning */
        bool        _is_pv = false;
        bool        _is_retry = false;
        bool        _is_singleton = false;
        bool        _is_terminal = false; /* debug only */
        bool        _multicut_allowed = true;
        bool        _null_move_allowed[2] = { true, true };
        RETRY       _retry_above_alpha = RETRY::None;
        bool        _retry_next = false;
        int         _double_ext = 0;
        int         _extension = 0; /* count pending fractional extensions */
        int         _fifty = 0;
        int         _full_depth_count = late_move_reduction_count();
        int         _mate_detected = 0;
        int         _pruned_count = 0;
        int         _prune_reason = 0; /* debug */

        HistoryPtr  _history;

        Move        _cutoff_move;   /* from current state to the next */
        Move        _move;          /* from parent to current state */
        BaseMove    _prev;          /* best move from previous iteration */
        BaseMove    _excluded;      /* singular extension search */

        State*      _state = nullptr;
        TT_Entry    _tt_entry;
        Square      _capture_square = Square::UNDEFINED;

        const ContextPtr& best() const { return _best; }
        void set_best(const ContextPtr& best) { best->_parent = this; _best = best; }

        static void cancel();
        static int  cpu_cores();

        bool        can_forward_prune() const;
        bool        can_prune() const;
        bool        can_prune_move(const Move&) const;
        bool        can_reduce();

        int64_t     check_time_and_update_nps(); /* return elapsed milliseconds */
        void        copy_move_state();

        int         depth() const { return _max_depth - _ply; }

        std::string epd() const;

        /* Static evaluation */
        score_t     _evaluate();    /* no repetitions, no fifty-moves rule */
        score_t     evaluate();     /* call _evaluate and do the above */
        score_t     evaluate_end();
        score_t     evaluate_material(bool with_piece_squares = true) const;
        int         eval_king_safety(int piece_count);
        int         eval_threats(int piece_count);

        void        extend();       /* fractional extensions */
        ContextPtr  first_move();
        score_t     futility_margin();

        bool        has_improved(score_t margin = 0) { return improvement() > margin; }
        bool        has_moves() { return _move_maker.has_moves(*this); }

        int         history_count(const Move&) const;
        float       history_score(const Move&) const;

        score_t     improvement();
        static void init();

        bool        is_beta_cutoff(const ContextPtr&, score_t);
        static bool is_cancelled() { return _cancel; }
        bool        is_capture() const { return state().capture_value != 0; }
        bool        is_check() const { return state().is_check(); }
        bool        is_counter_move(const Move&) const;
        bool        is_evasion() const;
        bool        is_extended() const;
        bool        is_last_move();
        bool        is_leftmost() const { return _ply == 0 || _leftmost; }
        bool        is_leaf(); /* treat as terminal node ? */
        bool        is_qsearch() const { return _ply > _max_depth; }
        bool        is_mate_bound() const;
        bool        is_null_move_ok(); /* ok to generate null move? */
        bool        is_null_move() const { return _is_null_move; }
        bool        is_promotion() const { return state().promotion; }
        bool        is_pv_node() const { return _is_pv; }
        bool        is_recapture() const;
        bool        is_reduced() const;
        bool        is_pvs_ok() const;
        int         is_repeated() const;
        bool        is_retry() const { return _is_retry; }
        bool        is_singleton() const { return _is_singleton; }
        int         iteration() const { ASSERT(_tt); return _tt->_iteration; }

        LMRAction   late_move_reduce(int move_count);
        static int  late_move_reduction_count();

        static void log_message(LogLevel, const std::string&, bool force = true);

        int64_t     nanosleep(int nanosec);
        ContextPtr  next(bool null_move = false, bool = false, score_t = 0);
        int         next_move_index() { return _move_maker.current(*this); }
        bool        on_next();

        void        reinitialize();
        int         rewind(int where = 0, bool reorder = false);

        void        set_counter_move(const Move& move) { _counter_move = move; }
        void        set_search_window(score_t, bool reset = false);

        static void set_time_limit_ms(int milliseconds);
        void        set_time_info(int time_left /* millisec */, int moves_left);
        void        set_tt(TranspositionTable* tt) { _tt = tt; }

        bool        should_verify_null_move() const;
        int         singular_margin() const;

        Color       turn() const { return state().turn; }

        const State& state() const { ASSERT(_state); return *_state; }
        TranspositionTable* get_tt() const { return _tt; }

        const MovesList& get_moves() const
        {
            return _move_maker.get_moves();
        }

        void set_initial_moves(const MovesList& moves)
        {
            _move_maker.set_initial_moves(moves);
        }

        /* retrieve PV from TT */
        const BaseMovesList& get_pv() const { return get_tt()->get_pv(); }

        /*
         * Python callbacks
         */
        static PyObject*    _engine; /* searcher instance */

        static std::string  (*_epd)(const State&);
        static void         (*_log_message)(int, const std::string&, bool);
        static void         (*_on_iter)(PyObject*, ContextPtr, score_t);
        static void         (*_on_next)(PyObject*, int64_t);
        static std::string  (*_pgn)(ContextPtr);
        static void         (*_print_state)(const State&);
        static void         (*_report)(PyObject*, std::vector<ContextPtr>&);
        static size_t       (*_vmem_avail)();

    private:
        const Move* get_next_move(score_t);
        bool has_cycle(const State&) const;

        int repeated_count(const State&) const;

        ContextPtr  _best; /* best search result */
        mutable int _can_prune = -1;
        State       _statebuf;
        bool        _leftmost = false;
        mutable int _repetitions = -1;
        Move        _counter_move;
        friend class MoveMaker;
        MoveMaker   _move_maker;

        static std::atomic_bool _cancel;
        static std::mutex _mutex; /* update time limit from another thread */

        static asize_t  _callback_count;
        static int      _time_limit; /* milliseconds */
        static time     _time_start;

        TranspositionTable* _tt = nullptr;
    };


    inline bool Context::can_forward_prune() const
    {
        if (_can_prune == -1)
        {
            _can_prune =
                ((_parent != nullptr)
                * !is_pv_node()
                * (_max_depth >= 6 || !is_qsearch())
                * !_excluded
                * (state().pushed_pawns_score <= 1)
                * !state().just_king_and_pawns())
                && (_parent->_mate_detected == 0 || _parent->_mate_detected % 2)
                && !is_check();
        }
        return _can_prune > 0;
    }


    inline bool Context::can_prune() const
    {
        ASSERT(_ply > 0);
        ASSERT(!is_null_move());
        ASSERT(_move);

        return !is_singleton()
             * !is_extended()
             * !is_pv_node()
             * !is_repeated()
             * _parent->can_prune_move(_move);
    }


    inline bool Context::can_prune_move(const Move& move) const
    {
        ASSERT(move && move._state && move != _move);

        return ((move != _tt_entry._hash_move)
                * (move._state->capture_value == 0)
                * (move.promotion() == chess::PieceType::NONE)
                * (move.from_square() != _capture_square)
                * !is_counter_move(move)
                * can_forward_prune())
            && !move._state->is_check();
    }


    inline bool Context::can_reduce()
    {
        ASSERT(!is_null_move());

        return ((_ply != 0)
                * !is_retry()
                * !is_singleton()
                * !is_extended()
                * (state().pushed_pawns_score <= 1))
            && (_move.from_square() != _parent->_capture_square)
            && !is_recapture()
            && !state().is_check();
    }


    static constexpr double PHI = 1.61803398875;

    template<std::size_t... I>
    constexpr std::array<int, sizeof ... (I)> margins(std::index_sequence<I...>)
    {
        return { static_cast<int>(50 * I + pow(I + PHI, M_E)) ... };
    }


    inline score_t Context::futility_margin()
    {
        if (_ply == 0 || !_futility_pruning || depth() < 1)
            return 0;

        /*
         * No need to check for futile moves when material is above alpha,
         * since no move can immediately result in decrease of material
         * (margin of one PAWN for piece-square evaluation).
         */
        if (evaluate_material() > std::max(_alpha, _score) + chess::WEIGHT[chess::PAWN])
            return 0;

        static const auto fp_margins = margins(std::make_index_sequence<PLY_MAX>{});

        return fp_margins[depth()] * can_forward_prune();
    }


    inline int Context::history_count(const Move& move) const
    {
        return _tt->historical_counters(state(), turn(), move).first;
    }


    inline float Context::history_score(const Move& move) const
    {
        ASSERT(_tt);
        ASSERT(move);
        ASSERT(move != _move);

        return COUNTER_MOVE_BONUS * is_counter_move(move)
            + _tt->history_score(_ply, state(), turn(), move);
    }


    inline bool Context::is_counter_move(const Move& move) const
    {
        return depth() >= COUNTER_MOVE_MIN_DEPTH && _counter_move == move;
    }


    inline bool Context::is_extended() const
    {
        ASSERT(_parent);
        return _max_depth > _parent->_max_depth;
    }


    inline bool Context::is_mate_bound() const
    {
        return _tt_entry._value >= MATE_HIGH && _tt_entry._depth >= depth();
    }


    inline bool Context::is_recapture() const
    {
        ASSERT(_parent);
        return _parent->is_capture() && (_move.to_square() == _parent->_move.to_square());
    }


    inline bool Context::is_reduced() const
    {
        ASSERT(_parent);
        return _max_depth < _parent->_max_depth;
    }


    inline bool Context::is_pvs_ok() const
    {
        return (_algorithm == NEGASCOUT) && !is_retry() && !is_leftmost();
    }


    /*
     * Check how much time is left, update NPS, call optional Python callback.
     */
    inline bool Context::on_next()
    {
        if (_tid == 0 && ++_callback_count >= CALLBACK_PERIOD)
        {
            _callback_count = 0; /* reset */

            auto millisec = check_time_and_update_nps();

            if (is_cancelled()) /* time is up? */
                return false;

            if (_on_next)
                cython_wrapper::call(_on_next, _engine, millisec);
        }

        return !is_cancelled();
    }


    inline int Context::rewind(int where, bool reorder)
    {
        return _move_maker.rewind(*this, where, reorder);
    }


    inline bool Context::should_verify_null_move() const
    {
        return depth() >= NULL_MOVE_MIN_VERIFICATION_DEPTH;
    }


    inline int Context::singular_margin() const
    {
        return SINGULAR_MARGIN * depth();
    }

} /* namespace */