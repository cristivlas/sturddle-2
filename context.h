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

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set> /* unordered_multiset */
#include "Python.h"
#include "config.h"
#include "search.h"
#include "utility.h"

constexpr auto FIRST_EXCHANGE_PLY = PLY_MAX;

/* Configuration API */
struct Param {
    int val = 0;
    int min_val;
    int max_val;
    std::string group;
    bool normal = false;
};

extern std::map<std::string, Param> _get_param_info();
extern void _set_param(const std::string&, int value, bool echo=false);
extern std::map<std::string, int> _get_params();


enum class LogLevel : int
{
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4
};


namespace search
{
    using time = std::chrono::time_point<std::chrono::steady_clock>;

    using Bitboard = chess::Bitboard;
    using Color = chess::Color;
    using Square = chess::Square;

    using HistoryPtr = std::unique_ptr<struct History>;

    using atomic_bool = std::atomic<bool>;
    using atomic_int = std::atomic<int>;
    using atomic_time = std::atomic<time>; /* sic */


    enum Algorithm : uint8_t
    {
        NEGAMAX,
        NEGASCOUT,
        MTDF,
    };


    /* For detecting repeated positions */
    struct History
    {
        History() = default;

        void emplace(const State& s) { _positions.emplace(s); }
        void clear() { _positions.clear(); }
        size_t count(const State& s) const { return _positions.count(s); }
        size_t size() const { return _positions.size(); }

        std::unordered_multiset<State, Hasher<State>> _positions;
        int _fifty = 0;
    };


    /*
     * Late-move pruning counts (initialization idea borrowed from Crafty)
     */
    template<std::size_t... I>
    static constexpr std::array<int, sizeof ... (I)> lmp(std::index_sequence<I...>)
    {
        return { static_cast<int>(LMP_BASE + pow(I + .5, 1.9)) ... };
    }


    static const auto LMP = lmp(std::make_index_sequence<PLY_MAX>{});


    /*
     * Helper for making and ordering moves
     */
    class MoveMaker
    {
        static constexpr int MAX_PHASE = 4;

    public:
        MoveMaker() = default;

        /* Return upperbound of legal moves, which may include pruned and quiet moves. */
        int count() const { return _count; }

        int current(Context&);

        void ensure_moves(Context&, bool order_root_moves = false);

        const Move* get_next_move(Context& ctxt, score_t futility = 0);

        bool group_quiet_moves() const { return _group_quiet_moves; }
        bool has_moves(Context&);
        bool have_skipped_moves() const { return _have_pruned_moves || _have_quiet_moves; }
        bool is_last(Context&);
        bool is_singleton(Context&);

        int rewind(Context&, int where, bool reorder);

        INLINE void swap(MoveMaker& other)
        {
            std::swap(_group_quiet_moves, other._group_quiet_moves);
            std::swap(_have_move, other._have_move);
            std::swap(_have_quiet_moves, other._have_quiet_moves);
            std::swap(_have_pruned_moves, other._have_pruned_moves);
            std::swap(_need_sort, other._need_sort);
            std::swap(_phase, other._phase);
            std::swap(_count, other._count);
            std::swap(_current, other._current);
            std::swap(_state_index, other._state_index);
        }

    private:
        bool can_late_move_prune(const Context& ctxt) const;

        void generate_unordered_moves(Context&, bool order_root_moves = false);
        const Move* get_move_at(Context& ctxt, int index, score_t futility = 0);

        void make_capture(Context&, Move&);

        template<bool LateMovePrune> bool make_move(Context&, Move&, score_t futility = 0);
        template<bool LateMovePrune> bool make_move(Context&, Move&, MoveOrder, float = 0);

        void remake_move(Context&, Move&);

        void mark_as_illegal(Move&);
        void mark_as_pruned(Context&, Move&);

        void order_moves(Context&, size_t start_at, score_t futility_margin);

        template<int Phase>
        void order_moves_phase(
            Context&,
            MovesList&,
            size_t  start_at,
            size_t  count,
            score_t futility_margin);

        void sort_moves(Context&, size_t start_at, size_t count);

        MoveMaker(const MoveMaker&) = delete;
        MoveMaker& operator=(const MoveMaker&) = delete;

        bool        _group_quiet_moves = false;
        bool        _have_move = false;
        bool        _have_quiet_moves = false;
        bool        _have_pruned_moves = false;
        bool        _need_sort = false;
        int8_t      _phase = 0; /* move ordering phase */
        int         _count = -1;
        int         _current = -1;
        size_t      _state_index = 0;
    };


    enum class LMRAction : int { None = 0, Ok, Prune };

    /* Reason for retrying */
    enum class RETRY : uint8_t { None = 0, Reduced, PVS };

    INLINE constexpr bool operator!(RETRY retry) { return retry == RETRY::None; }

    struct IterationInfo
    {
        score_t score;
        size_t nodes;
        double knps;
        int milliseconds;
    };


    struct TimeControl
    {
        int millisec[2];    /* time left until next time control for black/white */
        int increments[2];  /* time increments for black/white */
        int moves;          /* number of moves till next time control */
        score_t score;
        score_t delta;      /* score difference from last search */
    };


    enum class PruneReason : uint8_t
    {
        PRUNE_NONE,
        PRUNE_END_TABLES,
        PRUNE_LMP,
        PRUNE_FUTILITY,
        PRUNE_MULTICUT,
        PRUNE_RAZOR,
        PRUNE_REVERSE_FUTILITY,
        PRUNE_SINGULAR,
        PRUNE_TT,
    };


    using PV = std::vector<BaseMove>;

    /*
     * The context of a searched node.
     */
    struct Context
    {
        using ContextStack = std::array<struct ContextBuffer, PLY_MAX>;

        /* Note: stack beyond PLY_MAX is reserved for do_exchanges. */
        static constexpr int MAX_MOVE = PLY_MAX + 32;
        using MoveStack = std::array<MovesList, MAX_MOVE>;
        using StatePool = std::vector<State>;
        using StateStack = std::array<StatePool, PLY_MAX>;

        using Path = std::array<BaseMove, PV_PATH_MAX>;
        using TT_Result = search::HashTable::Result;

        friend class MoveMaker;

        Context() = default;
        ~Context() = default;

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        static void* operator new(size_t, void* p) { return p; }
        static void* operator new(size_t) = delete;
        static void operator delete(void*, size_t) noexcept = delete;

        /* parent move in the graph */
        Context*    _parent = nullptr;
        int16_t     _ply = 0;
        int16_t     _max_depth = 0;
        int16_t     _alpha = SCORE_MIN;
        int16_t     _beta = SCORE_MAX;
        int16_t     _score = SCORE_MIN; /* dynamic eval score */
        int16_t     _retry_beta = SCORE_MAX; /* NEGASCOUT only */
        mutable int _improvement = SCORE_MIN;

        Algorithm   _algorithm = Algorithm::MTDF;

        mutable int8_t _can_forward_prune = -1;
        mutable int8_t _repetitions = -1;

        Square      _capture_square = Square::UNDEFINED;
        bool        _futility_pruning = true;
        bool        _has_singleton = false;
        bool        _is_null_move = false; /* for null-move pruning */
        bool        _is_pv = false;
        bool        _is_retry = false;
        bool        _is_singleton = false;
        bool        _leftmost = false;
        bool        _multicut_allowed = MULTICUT;
        bool        _null_move_allowed[2] = { true, true };
        RETRY       _retry_above_alpha = RETRY::None;
        bool        _retry_next = false;
        int8_t      _nnue_prev_offs = 1; /* NNUE */

        uint8_t     _double_ext = 0;
        uint8_t     _path_len = 1;
        Path        _path;

        score_t     _eval = SCORE_MIN; /* static eval */
        score_t     _eval_raw = SCORE_MIN; /* unscaled _eval */

        int         _extension = 0; /* count pending fractional extensions */
        int         _fifty = 0;
        int         _mate_detected = 0;
        int         _pruned_count = 0;
        PruneReason _prune_reason = PruneReason::PRUNE_NONE;

        Move        _move;          /* from parent to current state */
        BaseMove    _best_move;
        BaseMove    _counter_move;
        BaseMove    _cutoff_move;   /* from current state to the next */
        BaseMove    _prev;          /* best move from previous iteration */
        BaseMove    _excluded;      /* singular extension search */

        State*      _state = nullptr;

        void        cache_scores(bool force_write /* bypass eviction strategy */ = false);

        static void cancel() { _cancel.store(true, std::memory_order_relaxed); }

        bool        can_forward_prune() const;

        template<bool PruneCaptures = false> bool can_prune() const;
        template<bool PruneCaptures = false> bool can_prune_move(const Move&) const;

        bool        can_reduce() const;

        bool        can_reuse_moves() const;

        int64_t     check_time_and_update_nps(int64_t* = nullptr); /* return elapsed milliseconds */

        static void clear_caches_and_stacks();

        Context*    clone(ContextBuffer&, int ply = 0) const;

        int         depth() const { return _max_depth - _ply; }

        static int64_t elapsed_milliseconds();

        void        ensure_prev_move();
        static void ensure_stacks();

        std::string epd() const { return epd(state()); }
        static std::string epd(const State&);

        /* Static evaluation */
        score_t     _evaluate();

        template<bool EvalCaptures = true> score_t evaluate();

        score_t     evaluate_material() const;

        score_t     eval(bool as_white, int depth, int millisec); /* testing and tuning */

        score_t     eval_nnue_raw(bool side_to_move_pov);
        void        eval_with_nnue();
        static void update_root_accumulators();

        score_t     static_eval() const; /* use TT value if available, eval material otherwise */

        void        extend(); /* fractional and other extensions */
        const Move* first_valid_move();

        score_t     futility_margin() const;

        INLINE bool has_improved() const { return improvement() > 0; }
        INLINE bool has_moves() { return _move_maker.has_moves(*this); }
        INLINE bool has_pruned_moves() const { return _pruned_count || _move_maker.have_skipped_moves(); }

        int         history_count(const Move&) const;
        float       history_score(const Move&) const;

        score_t     improvement() const;
        static void init();
        bool        is_beta_cutoff(Context*, score_t);
        static bool is_cancelled() { return _cancel.load(std::memory_order_acquire); }
        INLINE bool is_capture() const { return state().is_capture(); }
        INLINE bool is_check() const { return state().is_check(); }
        bool        is_counter_move(const Move&) const;
        bool        is_evasion() const;
        bool        is_extended() const;
        bool        is_last_move();
        INLINE bool is_leftmost() const { return is_root() || _leftmost; }
        bool        is_leaf(); /* treat as terminal node ? */
        bool        is_mate_bound() const;
        bool        is_null_move_ok() const; /* ok to generate null move? */
        INLINE bool is_null_move() const { return _is_null_move; }
        INLINE bool is_promotion() const { return _move.promotion(); }
        INLINE bool is_pv_node() const { return _is_pv; }
        bool        is_pvs_ok() const;
        INLINE bool is_leaf_extended() const { return _ply > _max_depth; }
        bool        is_recapture() const;
        bool        is_reduced() const;
        int         is_repeated() const;
        INLINE bool is_retry() const { return _is_retry; }
        INLINE bool is_root() const { return _ply == 0; }

        INLINE int  iteration() const { ASSERT(_tt); return _tt->_iteration; }

        LMRAction   late_move_reduce(int move_count, int64_t time_left);

        static void log_message(LogLevel, const std::string&, bool force = false);

        int         move_count() const { return _move_maker.count(); }
        int64_t     nanosleep(int nanosec);

        Context*    next(bool null_move, score_t, int& move_count, int64_t* time_left = nullptr);

        template<bool Construct = false> Context* next_ply() const;

        INLINE int  next_move_index() { return _move_maker.current(*this); }

        bool        on_next(int64_t*);
        INLINE int  piece_count() const { return state().piece_count(); }

        void        reset(bool force_reorder_moves = true, bool clear_best_move = true);
        int         repeated_count(const State&) const;

        int         rewind(int where = 0, bool reorder = false);
        INLINE void set_counter_move(const BaseMove& move) { _counter_move = move; }
        void        set_search_window(score_t score, score_t& prev_score);
        static void set_start_time();
        static void set_time_limit_ms(int milliseconds);
        void        set_time_ctrl(const TimeControl&);
        INLINE void set_tt(TranspositionTable* tt) { _tt = tt; }
        bool        should_verify_null_move() const;

        int         tid() const { return _tt ? _tt->_tid : 0; }
        static int  time_limit() { return _time_limit.load(std::memory_order_relaxed); }
        Color       turn() const { return state().turn; }

        INLINE TT_Result& tt_result() { return StorageView<TT_Result>::get(_state->tt_result, _state->has_tt_result); }
        INLINE const TT_Result& tt_result() const { return StorageView<TT_Result>::get(_state->tt_result, _state->has_tt_result); }

        INLINE TT_Entry& tt_entry() { return tt_result()._entry; }
        INLINE const TT_Entry& tt_entry() const { return tt_result()._entry; }

        INLINE const State& state() const { ASSERT(_state); return *_state; }
        INLINE TranspositionTable* get_tt() const { return _tt; }

        INLINE const MovesList& moves() const { return moves(tid(), _ply); }
        INLINE MovesList& moves() { return moves(tid(), _ply); }

        INLINE const PV& get_pv() const
        {
            auto& pv = _pvs[tid()];
            const auto size = std::min<size_t>(_path_len, _path.size());
            pv.resize(size);
            // off-by-one because PV[0] = root context move
            std::copy_if(_path.begin() + 1, _path.begin() + size, pv.begin() + 1, [](Move move) {return bool(move);});
            return pv;
        }

        /* buffers for generating and making moves */
        static MovesList& moves(int tid, int ply);
        static StatePool& states(int tid, int ply);

        static void set_syzygy_path(const std::string& path) { _syzygy_path = path; }
        static const std::string& syzygy_path() { return _syzygy_path; }
        static void set_tb_cardinality(int n) { _tb_cardinality = n; }
        static int tb_cardinality() { return _tb_cardinality.load(std::memory_order_relaxed); }

        /*
         * Python callbacks
         */
        static PyObject*    _engine; /* searcher instance */

        static bool         (*_book_init)(const std::string&);
        static BaseMove     (*_book_lookup)(const State&, bool);
        static std::string  (*_epd)(const State&);
        static void         (*_log_message)(int, const std::string&, bool);
        static void         (*_on_iter)(PyObject*, Context*, const IterationInfo*);
        static void         (*_on_move)(PyObject*, const std::string&, int);
        static void         (*_on_next)(PyObject*, int64_t);
        static std::string  (*_pgn)(Context*);
        static void         (*_print_state)(const State&, bool unicode);
        static void         (*_report)(PyObject*, std::vector<Context*>&);
        static void         (*_set_syzygy_path)(const std::string&);
        static bool         (*_tb_probe_wdl)(const State&, int*);
        static size_t       (*_vmem_avail)();

        static HistoryPtr   _history;

    private:
        const Move* get_next_move(score_t);
        bool has_cycle(const State&) const;

        void update_accumulators();

        MoveMaker           _move_maker;

        TranspositionTable* _tt = nullptr;

        /* search can be cancelled from any thread */
        static atomic_bool  _cancel;

        static size_t       _callback_count;
        static atomic_int   _time_limit; /* milliseconds */
        static atomic_time  _time_start;
        static std::string  _syzygy_path;
        static atomic_int   _tb_cardinality;

        static std::vector<ContextStack>    _context_stacks;
        static std::vector<MoveStack>       _move_stacks;
        static std::vector<StateStack>      _state_stacks;
        static std::vector<PV>              _pvs;
    };


    /*
     * Helper data structure for allocating context on ContextStacks
     */
    static_assert(std::is_trivially_destructible<Context>::value);

    struct alignas(64) ContextBuffer
    {
        uint8_t _mem [sizeof(Context)] = {};
        State _state; /* for null-move and clone() */
        bool _valid = false;

        INLINE Context* as_context(bool validate) { ASSERT(!validate || _valid); return reinterpret_cast<Context*>(&_mem[0]); }
        INLINE const Context* as_context(bool validate) const { ASSERT(!validate || _valid); return reinterpret_cast<const Context*>(&_mem[0]); }
    };


    template<bool Debug = false>
    int do_exchanges(const State&, Bitboard, int tid, int ply = FIRST_EXCHANGE_PLY);

    extern score_t eval_captures(Context& ctxt, score_t);


    static int INLINE capture_gain(const State& state, const State& next_state, const BaseMove& move)
    {
        ASSERT(next_state.is_capture());
        const auto state_eval = state.eval_lazy();
        const auto adjust = next_state.piece_value_adjustment(next_state.capture_type);
        return (next_state.eval_apply_delta(move, state) - state_eval + adjust) * SIGN[state.turn];
    }


    /*
     * Evaluate same square exchanges. Called by make_captures.
     */
    template<bool StaticExchangeEvaluation>
    INLINE score_t eval_exchanges(int tid, const Move& move)
    {
        score_t val = 0;

        if (move)
        {
            ASSERT(move._state);
            ASSERT(move._state->piece_type_at(move.to_square()));

            if constexpr(StaticExchangeEvaluation)
            {
                /* Approximate without playing the moves. */
                val = estimate_static_exchanges(*move._state, move._state->turn, move.to_square());
            }
            else
            {
                auto mask = chess::BB_SQUARES[move.to_square()];
                val = do_exchanges<DEBUG_CAPTURES != 0>(*move._state, mask, tid);
            }
        }
        return val;
    }


    /* Evaluate material from the point of view of the side that just moved.
     * Include piece-square values if USE_PIECE_SQUARE_TABLES defined as true (see common.h)
     */
    score_t eval_material_for_side_that_moved(const State& state, const State* prev, const BaseMove& move);


    INLINE void incremental_update(Move& move, const Context& ctxt)
    {
        ASSERT(move._state);
        ASSERT(ctxt._ply < PLY_MAX - 1);

        if (move._state->simple_score == State::UNKNOWN_SCORE)
        {
            move._state->eval_apply_delta(move, ctxt.state());
        }

        /* post-condition */
        ASSERT(move._state->simple_score == move._state->eval_simple());
    }


    INLINE bool is_direct_check(const Move& move)
    {
        ASSERT(move);
        ASSERT(move._state);

        const auto& state = *move._state;
        if (state.attacks_mask(move.to_square(), state.occupied()) & state.kings & state.occupied_co(state.turn))
        {
            ASSERT(state.is_check());
            return true;
        }
        return false;
    }


    INLINE bool is_quiet(const Move& move)
    {
        return move.promotion() != chess::PieceType::QUEEN /* ignore under-promotions */
            && !move._state->is_capture()
            && move._state->pushed_pawns_score <= 1
            && !move._state->is_check();
    }


    INLINE void copy_search_path(const Context& from_ctxt, Context& to_ctxt)
    {
        const auto size = std::min<size_t>(from_ctxt._path_len, PV_PATH_MAX);
        std::copy_n(from_ctxt._path.begin(), size, to_ctxt._path.begin());
        to_ctxt._path_len = from_ctxt._path_len;
    }


    INLINE bool Context::can_forward_prune() const
    {
        if (_can_forward_prune == -1)
        {
            _can_forward_prune =
                (_parent != nullptr)
                && !is_pv_node()
                && !_excluded
                && (state().pushed_pawns_score <= 2)
                && !state().just_king_and_pawns()
                && (_parent->_mate_detected == 0 || _parent->_mate_detected % 2)
                && !is_check();
        }
        return _can_forward_prune > 0;
    }


    template<bool PruneCaptures>
    INLINE bool Context::can_prune() const
    {
        ASSERT(_ply > 0);
        ASSERT(!is_null_move());
        ASSERT(_move);
        ASSERT(_move._state);

        return !is_extended()
            && !is_pv_node()
         // && !is_retry()
            && !is_repeated()
            && _parent->can_prune_move<PruneCaptures>(_move);
    }


    template<bool PruneCaptures>
    INLINE bool Context::can_prune_move(const Move& move) const
    {
        ASSERT(move && move._state && move != _move);

        return (move != tt_entry()._best_move)
            && (move != tt_entry()._hash_move)
            && (PruneCaptures || !move._state->is_capture())
            && (move.promotion() == chess::PieceType::NONE)
            && (move.from_square() != _capture_square)
            && can_forward_prune()
            && !move._state->is_check();
    }


    INLINE bool Context::can_reduce() const
    {
        ASSERT(!is_null_move());

        return !is_leftmost()
            && !is_retry()
            && (state().pushed_pawns_score <= 1)
            && !is_extended()
            && (_move.from_square() != _parent->_capture_square)
            && !is_recapture()
            && !is_check();
    }


    INLINE bool Context::can_reuse_moves() const
    {
        return !_move_maker.have_skipped_moves() && !_move_maker.group_quiet_moves();
    }


    /* static */ INLINE int64_t Context::elapsed_milliseconds()
    {
        const auto now = std::chrono::steady_clock::now();

        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now - _time_start.load(std::memory_order_relaxed)
        ).count();
    }


    INLINE int eval_fuzz()
    {
    #if EVAL_FUZZ_ENABLED
        return EVAL_FUZZ ? random_int(-EVAL_FUZZ, EVAL_FUZZ) : 0;
    #else
        return 0;
    #endif /* EVAL_FUZZ_ENABLED */
    }


    template <typename F = std::function<score_t()>>
    INLINE score_t eval_insufficient_material(const State& state, score_t eval=SCORE_MIN, F f=[]{ return SCORE_MIN; })
    {
        if (state.is_endgame() && state.has_insufficient_material(state.turn))
        {
            if (state.has_insufficient_material(!state.turn))
            {
                eval = 0; /* neither side can win */
            }
            else
            {
                eval = std::min<score_t>(eval, 0); /* cannot do better than draw */
            }
        }
        else
        {
            eval = f();
        }
        return eval;
    }


    /*
     * Use value from the TT if available, else use material evaluation.
     */
    INLINE score_t Context::static_eval() const
    {
        if (is_valid(_eval))
            return _eval;

        if (is_valid(tt_entry()._value) && tt_entry()._depth >= depth())
            return tt_entry()._value;

        return evaluate_material();
    }


#if !WITH_NNUE
    INLINE void search::Context::eval_with_nnue() {}
    INLINE score_t search::Context::eval_nnue_raw(bool) { return 0; }
    INLINE void search::Context::update_root_accumulators() {}
#endif /* !WITH_NNUE */


    template<bool EvalCaptures> INLINE score_t Context::evaluate()
    {
        ASSERT(_fifty < 100);
        ASSERT(is_root() || !is_repeated());

        ++_tt->_eval_count;

        auto score = _evaluate();

        ASSERT(score > SCORE_MIN);
        ASSERT(score < SCORE_MAX);

        if constexpr(EvalCaptures)
        {
            /* Captures may not make a difference for large score gaps. */
            if (abs(score) < CAPTURES_THRESHOLD)
            {
                score += eval_captures(*this, score);

                ASSERT(score > SCORE_MIN);
                ASSERT(score < SCORE_MAX);
            }
        }
        return score;
    }


    INLINE score_t Context::evaluate_material() const
    {
        /*
         * Flip it from the pov of the side that moved
         * to the perspective of the side-to-move.
         */
        return -eval_material_for_side_that_moved(*_state, _parent ? _parent->_state : nullptr, _move);
    }


    /*
     * Futility pruning margins.
     */
    template<std::size_t... I>
    static constexpr std::array<int, sizeof ... (I)> margins(std::index_sequence<I...>)
    {
    #if 0
        return { static_cast<int>(75 * I + pow(I, 1.99)) ... };
    #else
        return { static_cast<int>(std::min(75.0 * I + pow(I, 1.99), 1289.0 + 200.0 * log(I))) ... };
    #endif
    }


    INLINE score_t Context::futility_margin() const
    {
        if (is_root() || !_futility_pruning || depth() < 1)
            return 0;

        static const auto fp_margins = []() {
            const auto m = margins(std::make_index_sequence<PLY_MAX>{});
            return m;
        } ();
        return fp_margins[depth()] * can_forward_prune();
    }


    INLINE const Move* Context::get_next_move(score_t futility)
    {
        auto move = _move_maker.get_next_move(*this, futility);

        if (move && *move == _excluded)
        {
            move = _move_maker.get_next_move(*this, futility);
        }
        return move;
    }


    INLINE int Context::history_count(const Move& move) const
    {
        return _tt->historical_counters(state(), turn(), move).first;
    }


    INLINE float Context::history_score(const Move& move) const
    {
        ASSERT(_tt);
        ASSERT(move);
        ASSERT(move != _move);

        const auto score = _tt->history_score(_ply, state(), turn(), move);
        return score + COUNTER_MOVE_BONUS * is_counter_move(move);
    }


    /*
     * Improvement for the side that just moved.
     */
    INLINE score_t Context::improvement() const
    {
        if (_improvement < 0)
        {
            if (_ply < 2 || _excluded || is_promotion())
            {
                _improvement = 0;
            }
            else
            {
                const auto prev = _parent->_parent;
                const auto eval = static_eval();
                const auto prev_eval = prev->static_eval();

                if (abs(eval) < MATE_HIGH && abs(prev_eval) < MATE_HIGH)
                {
                    _improvement = std::max(0, prev_eval - eval);
                }
                else
                {
                    const auto gg_parent = prev->_parent;
                    const auto gg_parent_state = gg_parent ? gg_parent->_state : nullptr;

                    _improvement = std::max(0,
                          eval_material_for_side_that_moved(*_state, _parent->_state, _move)
                        - eval_material_for_side_that_moved(*prev->_state, gg_parent_state, prev->_move));
                }
            }
        }

        return _improvement;
    }


    INLINE bool Context::is_counter_move(const Move& move) const
    {
        return _counter_move == move;
    }


    INLINE bool Context::is_evasion() const
    {
        return _parent && _parent->is_check();
    }


    INLINE bool Context::is_extended() const
    {
        ASSERT(_parent);
        return _max_depth > _parent->_max_depth;
    }


    INLINE bool Context::is_mate_bound() const
    {
        return tt_entry()._value >= MATE_HIGH && tt_entry()._depth >= depth();
    }


    /*
     * Ok to generate a null-move?
     */
    INLINE bool Context::is_null_move_ok() const
    {
        if (is_root()
            || depth() < NULL_MOVE_MIN_DEPTH
            || _null_move_allowed[turn()] == false
            || _excluded
            || is_null_move() /* consecutive null moves are not allowed */
            || is_pv_node()
            || is_mate_bound()
            || is_repeated()
            || is_check()
            || state().just_king_and_pawns()
           )
            return false;

        return static_eval() >= _beta - NULL_MOVE_DEPTH_WEIGHT * depth() + NULL_MOVE_MARGIN;
    }


    INLINE bool Context::is_recapture() const
    {
        ASSERT(_parent);
        return _parent->is_capture() && (_move.to_square() == _parent->_move.to_square());
    }


    INLINE bool Context::is_reduced() const
    {
        ASSERT(_parent);
        return _max_depth < _parent->_max_depth;
    }


    INLINE bool Context::has_cycle(const State& state) const
    {
        const auto hash = state.hash();

        for (auto ctxt = _parent; ctxt; ctxt = ctxt->_parent)
        {
            if (hash == ctxt->state().hash() && state == ctxt->state())
            {
                return true;
            }
        }
        return false;
    }


    INLINE int Context::repeated_count(const State& state) const
    {
        ASSERT(_history);
        return int(_history->count(state)) + has_cycle(state);
    }


    INLINE int Context::is_repeated() const
    {
        if (_repetitions < 0)
        {
            ASSERT(_history);
            _repetitions = repeated_count(state());

            ASSERT(_repetitions >= 0);
        }
        return _repetitions;
    }


    INLINE bool Context::is_pvs_ok() const
    {
        return (_algorithm == NEGASCOUT) && !is_retry() && !is_leftmost();
    }


    /*
     * Reduction formula based on ideas from SF and others.
     */
    INLINE int null_move_reduction(const Context& ctxt)
    {
        return NULL_MOVE_REDUCTION_BASE /* base reduction */
            + ctxt.depth() / NULL_MOVE_REDUCTION_DEPTH_DIV
            + std::min(ctxt.depth() / 2, (ctxt.static_eval() - ctxt._beta) / NULL_MOVE_REDUCTION_DIV);
    }


    /*
     * Get the next move and wrap it into a Context object.
     */
    INLINE Context* Context::next(bool make_null_move, score_t futility, int& move_count, int64_t* time_left)
    {
        ASSERT(_alpha < _beta);

        const bool retry = _retry_next;
        if (retry)
        {
            ASSERT(move_count > 0);
            --move_count;
        }
        _retry_next = false;

        if (!on_next(time_left))
            return nullptr;

        /* null move must be tried before actual moves */
        ASSERT(!make_null_move || move_count == 0);

        const Move* move = nullptr;

        if (!make_null_move)
            if ((move = get_next_move(futility)) == nullptr)
                return nullptr;

        ASSERT(make_null_move || move->_state);
        ASSERT(make_null_move || move->_group != MoveOrder::UNDEFINED);
        ASSERT(make_null_move || move->_group < MoveOrder::UNORDERED_MOVES);

        /* Save previously generated moves for reuse on retry */
        MoveMaker temp;
        if (retry)
        {
            const auto ctxt = next_ply<false>();
            if (ctxt->move_count() >= 0)
            {
                if (ctxt->can_reuse_moves())
                    ctxt->_move_maker.swap(temp);
                else
                    ctxt->cache_scores(true /* force write */);
            }
        }

        auto ctxt = next_ply<true>(); /* Construct new context */

        copy_search_path(*this, *ctxt);

        if (move)
        {
            ASSERT(move->_state);
            ASSERT(ctxt->_is_null_move == false);

            ctxt->_move = *move;
            ctxt->_state = move->_state;
            ctxt->_leftmost = is_leftmost() && next_move_index() == 1;

            if (_path_len >= _ply + 1 && _ply + 1 < PV_PATH_MAX)
            {
                /* Add move to search path */
                ctxt->_path[_ply + 1] = *move;
                ctxt->_path_len = _ply + 2;
            }

        #if REPORT_CURRENT_MOVE
            /* Report (main thread only) the move being searched from the root. */
            if (is_root()
                && tid() == 0
                && _on_move
                && (_tt->_nodes % 1000) <= size_t(move_count)
                && time_limit() > 250
               )
                (*_on_move)(_engine, move->uci(), move_count + 1);
        #endif /* REPORT_CURRENT_MOVE */
        }
        else
        {
            ASSERT(make_null_move);
            ASSERT(!ctxt->_move);
            ASSERT(ctxt->_state);

            state().clone_into(*ctxt->_state);
            ctxt->_state->_check = { 0, 0 };
            flip(ctxt->_state->turn);
            ctxt->_is_null_move = true;
        }

        ctxt->_algorithm = _algorithm;
        ctxt->_parent = this;
        ctxt->_max_depth = _max_depth;
        ctxt->_ply = _ply + 1;
        ctxt->_double_ext = _double_ext;
        ctxt->_extension = _extension;
        ctxt->_is_retry = retry;
        if (is_root())
        {
            ctxt->_is_singleton = !ctxt->is_null_move() && _move_maker.is_singleton(*this);
            _has_singleton = ctxt->_is_singleton;
        }
        ctxt->_futility_pruning = _futility_pruning && FUTILITY_PRUNING;
        ctxt->_multicut_allowed = _multicut_allowed && MULTICUT;

        for (auto side : { chess::BLACK, chess::WHITE })
            ctxt->_null_move_allowed[side] = _null_move_allowed[side] && (NULL_MOVE_REDUCTION_BASE > 0);

        ctxt->_tt = _tt;
        ctxt->_alpha = -_beta;

        if (ctxt->is_null_move())
        {
            ASSERT(make_null_move);
            ASSERT(NULL_MOVE_REDUCTION_BASE > 0);

            ctxt->_beta = -_beta + 1;

        #if ADAPTIVE_NULL_MOVE
            ASSERT(depth() >= 0);

            const auto reduction = std::min(depth(), std::max(null_move_reduction(*this), 0));
            ASSERT(reduction >= 0);

            ctxt->_max_depth -= reduction;

            /*
             * Allow the null-move to descend one ply into qsearch, to get the
             * mate detection side-effect; forward pruning is disabled when mate
             * is detected, thus enabling the engine to understand better quiet
             * moves that involve material sacrifices.
             */
            ASSERT(ctxt->depth() >= -1);
        #else
            if (ctxt->depth() > 7)
                --ctxt->_max_depth; /* reduce more */

            ctxt->_max_depth -= NULL_MOVE_REDUCTION;

        #endif /* !ADAPTIVE_NULL_MOVE */
        }
        else
        {
            ASSERT(move);

            ctxt->_beta = -_alpha;

            if (ctxt->is_pvs_ok())
            {
                ctxt->_alpha = -_alpha - 1;
                ctxt->_retry_above_alpha = RETRY::PVS;
                ASSERT(ctxt->_alpha == ctxt->_beta - 1);
            }
            else if (ctxt->is_retry() && _retry_beta < _beta)
            {
                ASSERT(_algorithm == Algorithm::NEGASCOUT);
                ctxt->_beta = _retry_beta;
                _retry_beta = SCORE_MAX;
            }

            /*
             * https://en.wikipedia.org/wiki/Fifty-move_rule
             */
            if constexpr(FIFTY_MOVES_RULE)
            {
                if (ctxt->is_capture())
                    ctxt->_fifty = 0;
                else if (state().pawns & chess::BB_SQUARES[move->from_square()])
                    ctxt->_fifty = 0;
                else
                    ctxt->_fifty = (is_root() ? _history->_fifty : _fifty) + 1;
            }

            if (temp.count() >= 0)
            {
                /* Reuse previously generated moves */
                ctxt->_move_maker.swap(temp);
                ASSERT(temp.count() == -1);

                ctxt->rewind(0);
            }
        }

        ASSERT(ctxt->_alpha < ctxt->_beta);
        ASSERT(ctxt->_alpha >= ctxt->_score);
        ASSERT(ctxt->_score == SCORE_MIN);

        return ctxt;
    }


    /*
     * Check how much time is left, update NPS, call optional Python callback.
     */
    INLINE bool Context::on_next(int64_t* time_left)
    {
        if (tid() == 0 && ++_callback_count >= CALLBACK_PERIOD)
        {
            _callback_count = 0; /* reset */
            const auto millisec = check_time_and_update_nps(time_left);

            if (millisec < 0) /* time is up? */
                return false;

            if (_on_next)
                cython_wrapper::call(_on_next, _engine, millisec);
        }

        return !is_cancelled();
    }


    INLINE int Context::rewind(int where, bool reorder)
    {
        return _move_maker.rewind(*this, where, reorder);
    }


    INLINE bool Context::should_verify_null_move() const
    {
    #if 0
        return depth() >= NULL_MOVE_MIN_VERIFICATION_DEPTH;
    #else
        return depth() >= std::max(NULL_MOVE_MIN_VERIFICATION_DEPTH, _max_depth - 8);
    #endif
    }


    template<bool Construct> INLINE Context* Context::next_ply() const
    {
        ASSERT(_ply < PLY_MAX);

        auto& buffer = _context_stacks[tid()][_ply];

        if constexpr(Construct)
        {
            auto ctxt = new (buffer.as_context(false)) Context;
            buffer._valid = true;
            ctxt->_state = &buffer._state;

            return ctxt;
        }
        else
        {
            return buffer.as_context(true);
        }
    }


    /* static */ INLINE MovesList& Context::moves(int tid, int ply)
    {
        ASSERT(ply >= 0);
        ASSERT(size_t(ply) < MAX_MOVE);

        return _move_stacks[tid][ply];
    }


    /* static */ INLINE Context::StatePool& Context::states(int tid, int ply)
    {
        ASSERT(ply >= 0);
        ASSERT(size_t(ply) < PLY_MAX);

        return _state_stacks[tid][ply];
    }


    /* static */ INLINE void Context::set_start_time()
    {
        _cancel = false;
        _time_start.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
        _callback_count = 0;
    }


    /* static */ INLINE void Context::set_time_limit_ms(int time_limit)
    {
        set_start_time();
        _time_limit.store(time_limit, std::memory_order_relaxed);
    }


    INLINE void Context::set_time_ctrl(const TimeControl& ctrl)
    {
        // Margin for OS context-switching, I/O overhead, etc.
        constexpr int MAX_SAFETY_MARGIN = 10;

        const auto side_to_move = turn();
        const auto millisec = std::max(0, ctrl.millisec[side_to_move]);
        const auto bonus = std::max(0, ctrl.increments[side_to_move]);

        int moves = ctrl.moves; // Moves left until next time control
        if (moves == 0)
        {
            const int moves_played = int(_history->size() / 2); // plies -> moves
            const int estimated_moves_left = AVERAGE_MOVES_PER_GAME - moves_played;

            if (bonus >= millisec - MAX_SAFETY_MARGIN)
            {
                moves = 1; // bonus dominates, play move-by-move
            }
            else
            {
                moves = std::max(AVERAGE_MOVES_PER_GAME / (2 + 2 * int(bonus > 0)), estimated_moves_left);
            }
        }

        int time_limit = std::max(millisec / moves, std::min(millisec, bonus));

        const int margin = std::min(MAX_SAFETY_MARGIN, time_limit / 15);
        time_limit = std::max(1, time_limit - margin);

        // DEBUG
        // std::cout << "info string movestogo " << moves << " time_limit " << time_limit << std::endl;

        _time_limit.store(time_limit, std::memory_order_relaxed);
    }


    INLINE bool MoveMaker::can_late_move_prune(const Context& ctxt) const
    {
        ASSERT(_phase > 1);

        return ctxt.depth() > 1 /* do not LMP leaf nodes */
            && _current >= LMP[ctxt.depth() - 1]
            && ctxt.can_forward_prune();
    }


    INLINE int MoveMaker::current(Context& ctxt)
    {
        ensure_moves(ctxt);
        return _current;
    }


    INLINE void MoveMaker::ensure_moves(Context& ctxt, bool order_root_moves)
    {
        ASSERT(ctxt.iteration());
        if (_count < 0)
        {
            ASSERT(_current < 0);
            ASSERT(_phase == 0);

            generate_unordered_moves(ctxt, order_root_moves);

            ASSERT(_count >= 0);
            ASSERT(_current >= 0);
        }
    }


    INLINE const Move* MoveMaker::get_next_move(Context& ctxt, score_t futility)
    {
        ensure_moves(ctxt);

        /* ensure_moves post-condition */
        ASSERT(_current <= _count);

        auto move = get_move_at(ctxt, _current, futility);

        if (move)
        {
            ++_current;
        }
        else
        {
            ASSERT(_current == _count || ctxt.moves()[_current]._group >= PRUNED_MOVES);
            if (_count >= 0)
                ctxt.cache_scores();
        }
        return move;
    }


    INLINE const Move* MoveMaker::get_move_at(Context& ctxt, int index, score_t futility)
    {
        ensure_moves(ctxt);

        if (index >= _count)
        {
            return nullptr;
        }

        auto& moves_list = ctxt.moves();
        ASSERT(!moves_list.empty());

        auto move = &moves_list[index];
        ASSERT(move->_group != MoveOrder::UNDEFINED);

        while (move->_group == MoveOrder::UNORDERED_MOVES)
        {
            if (_phase > 2 && can_late_move_prune(ctxt))
            {
                mark_as_pruned(ctxt, *move);
                return nullptr;
            }

            order_moves(ctxt, index, futility);
            move = &moves_list[index];
        }

        ASSERT(move->_group != MoveOrder::UNORDERED_MOVES);

        if (move->_group >= MoveOrder::PRUNED_MOVES)
        {
            ASSERT(index <= _count);
            move = nullptr;
        }

        return move;
    }


    INLINE bool MoveMaker::has_moves(Context& ctxt)
    {
        ensure_moves(ctxt);
        ASSERT(_count >= 0);

        return (_count > 0) && get_move_at(ctxt, 0);
    }


    INLINE bool MoveMaker::is_last(Context& ctxt)
    {
        ensure_moves(ctxt);
        ASSERT(_count >= 0);

        return _current >= _count || !get_move_at(ctxt, _current);
    }


    /*
     * Is the current move the only available move?
     */
    INLINE bool MoveMaker::is_singleton(Context& ctxt)
    {
        ASSERT(_current > 0);

        if (_current > 1 || ctxt._pruned_count || have_skipped_moves())
            return false;

        ASSERT(_current == 1);

        if (_count == 1)
        {
        #if !defined(NO_ASSERT)
            /* checked for skipped moves above, make sure there aren't any */
            const auto& all_moves = ctxt.moves();
            for (const auto& move : all_moves)
            {
                ASSERT(move._group != MoveOrder::QUIET_MOVES);
                ASSERT(move._group != MoveOrder::PRUNED_MOVES);
            }
        #endif /* !(NO_ASSERT) */
            return true;
        }
        /*
         * get_move_at() is expensive;
         * use only if moving out of check.
         */
        return ctxt.is_check() && !get_move_at(ctxt, _current) && !have_skipped_moves();
    }


    INLINE void MoveMaker::make_capture(Context& ctxt, Move& move)
    {
        /* captures of the last piece moved by the opponent are handled separately */
        ASSERT(!ctxt._move || move.to_square() != ctxt._move.to_square());

        ASSERT(is_valid(ctxt._eval));

        if (   move._old_group == MoveOrder::LOSING_CAPTURES
            || move._old_group == MoveOrder::WINNING_CAPTURES
            || move._old_group == MoveOrder::EQUAL_CAPTURES)
        {
            /* Use values from before rewind / reorder */
            remake_move(ctxt, move);
        }
        else if (make_move<false>(ctxt, move))
        {
            ASSERT(move._state->is_capture());

            const auto gain = capture_gain(ctxt.state(), *move._state, move) - eval_exchanges<true>(ctxt.tid(), move);

            if (SEE_PRUNING
                && gain < SEE_PRUNING_MARGIN
                && !ctxt.is_root()
                && ctxt.depth() > 0
                && ctxt.depth() <= SEE_PRUNING_DEPTH
                && !ctxt.is_check())
            {
                mark_as_pruned(ctxt, move);
            }
            else
            {
                if (gain < 0)
                {
                    move._group = MoveOrder::LOSING_CAPTURES;
                }
                else
                {
                    static_assert(MoveOrder::WINNING_CAPTURES + 1 == MoveOrder::EQUAL_CAPTURES);
                    move._group = MoveOrder::WINNING_CAPTURES + (gain == 0);
                }

                move._score = gain;
            }
        }
    }


    /*
     * Return false if the move is not legal, or pruned.
     */
    template<bool LateMovePrune>
    INLINE bool MoveMaker::make_move(Context& ctxt, Move& move, score_t futility)
    {
        ASSERT(move);
        ASSERT(move._group == MoveOrder::UNORDERED_MOVES);

        _need_sort = true;

        /* Capturing the king is an illegal move (Louis XV?) */
        ASSERT((ctxt.state().kings & chess::BB_SQUARES[move.to_square()]) == chess::BB_EMPTY);

        /* Check the old group (saved to cache or from before rewinding) */
        /* NB: do not filter out old pruned moves, they are path-dependent. */
        if (move._old_group == MoveOrder::ILLEGAL_MOVES)
        {
            mark_as_illegal(move);
            return false;
        }
    #if GROUP_QUIET_MOVES
        else if (move._old_group == MoveOrder::QUIET_MOVES && _group_quiet_moves)
        {
            _have_quiet_moves = true;
            move._group = MoveOrder::QUIET_MOVES;
            return false;
        }
    #endif /* GROUP_QUIET_MOVES */

        if (move._state == nullptr)
        {
            /* Late-move prune before making the move. */
            if constexpr(LateMovePrune)
            {
                if (can_late_move_prune(ctxt))
                {
                    mark_as_pruned(ctxt, move);
                    return false;
                }
            }

            /* Ensure that there's a board state associated with the move. */
            ASSERT(_state_index < Context::states(ctxt.tid(), ctxt._ply).size());

            move._state = &Context::states(ctxt.tid(), ctxt._ply)[_state_index++];
        }
        /* Check legality in case this was a quiet, or previously pruned move */
        else if ((move._old_group == MoveOrder::UNDEFINED || move._old_group >= MoveOrder::UNORDERED_MOVES)
            && move._state->is_check(!move._state->turn))
        {
            mark_as_illegal(move);
            return false;
        }
        else
        {
            if constexpr(COUNT_VALID_MOVES_AS_NODES)
                ++ctxt.get_tt()->_nodes;

            return (_have_move = true);
        }

        ctxt.state().clone_into(*move._state);
        ASSERT(!move._state->is_capture());

        move._state->apply_move(move);

    #if GROUP_QUIET_MOVES
        if (_group_quiet_moves && is_quiet(move))
        {
            _have_quiet_moves = true;
            move._group = MoveOrder::QUIET_MOVES;
            return false;
        }
    #endif /* GROUP_QUIET_MOVES */

        incremental_update(move, ctxt);

        /* Futility pruning (1st pass, 2nd pass done in search) */
        /* Prune after making the move (state is needed for simple eval). */

        if (futility > 0)
        {
            /* The futility margin is calculated after at least one move has been searched. */
            ASSERT(_current > 0);

            const auto val = futility + eval_material_for_side_that_moved(*move._state, ctxt._state, move);

            if (val < ctxt._alpha && ctxt.can_prune_move<true>(move))
            {
                if constexpr(EXTRA_STATS)
                    ++ctxt.get_tt()->_futility_prune_count;

                mark_as_pruned(ctxt, move);
                return false;
            }
        }

        if (move._old_group == MoveOrder::UNDEFINED || move._old_group >= MoveOrder::UNORDERED_MOVES)
        {
            if (move._state->is_check(ctxt.turn()))
            {
                mark_as_illegal(move); /* can't leave the king in check */
                return false;
            }
        }

        /* consistency check */
        ASSERT((move._state->is_capture()) == ctxt.state().is_capture(move));

        if constexpr(COUNT_VALID_MOVES_AS_NODES)
            ++ctxt.get_tt()->_nodes;

        return (_have_move = true);
    }


    template<bool LateMovePrune>
    INLINE bool MoveMaker::make_move(Context& ctxt, Move& move, MoveOrder group, float score)
    {
        if (!make_move<LateMovePrune>(ctxt, move))
        {
            ASSERT(move._group >= MoveOrder::PRUNED_MOVES);
            return false;
        }
        ASSERT(move._group == MoveOrder::UNORDERED_MOVES);
        move._group = group;
        move._score = score;

        return true;
    }


    /* Remake previously rewound, or cached move. */
    INLINE void MoveMaker::remake_move(Context& ctxt, Move& move)
    {
        if (move._state == nullptr)
        {
            ASSERT(_state_index < Context::states(ctxt.tid(), ctxt._ply).size());
            move._state = &Context::states(ctxt.tid(), ctxt._ply)[_state_index++];

            ctxt.state().clone_into(*move._state);
            ASSERT(!move._state->is_capture());

            move._state->apply_move(move);
        }

        _need_sort = true;
        _have_move = true;

        move._group = static_cast<MoveOrder>(move._old_group);
        move._score = move._old_score;

        if constexpr(COUNT_VALID_MOVES_AS_NODES)
        {
            ++ctxt.get_tt()->_nodes;
        }
    }


    INLINE void MoveMaker::mark_as_illegal(Move& move)
    {
        move._group = MoveOrder::ILLEGAL_MOVES;
        ASSERT(_count > 0);
        --_count;
    }


    INLINE void MoveMaker::mark_as_pruned(Context& ctxt, Move& move)
    {
        ASSERT(!ctxt.is_root());
        move._group = MoveOrder::PRUNED_MOVES;
        _have_pruned_moves = true;
        ++ctxt._pruned_count;
    }


    /*
     * Sort moves ascending by group, descending by score within each group.
     */
    INLINE bool compare_moves_gt(const Move& lhs, const Move& rhs)
    {
        return (lhs._group == rhs._group && lhs._score > rhs._score) || (lhs._group < rhs._group);
    }


    INLINE bool compare_moves_ge(const Move& lhs, const Move& rhs)
    {
        return !compare_moves_gt(rhs, lhs);
    }


    INLINE void MoveMaker::sort_moves(Context& ctxt, size_t start_at, size_t count)
    {
        ASSERT(start_at < ctxt.moves().size());
        ASSERT(count <= ctxt.moves().size());

        auto& moves_list = ctxt.moves();
        const auto first = moves_list.begin() + start_at;
        const auto last = moves_list.begin() + count;

        insertion_sort(first, last, compare_moves_gt);

        _need_sort = false;
    }
} /* namespace search */


score_t eval(const std::string& epd, bool as_side_to_move, int depth = 0, int millis = -1);


/* C++ implementation of UCI protocol if NATIVE_UCI is defined,
 * or just a stub otherwise (and the uci.pyx impl is used instead).
 * See uci_native.cpp for details.
 */
void uci_loop(std::unordered_map<std::string, std::string> params);


/* Make uci_loop accessible from Python */
INLINE void _uci_loop(std::unordered_map<std::string, std::string> params) noexcept
{
    cython_wrapper::call_nogil(uci_loop, params);
}


namespace nnue
{
    /**
     * Evaluate FEN from White's point of view, for testing.
     */
#if WITH_NNUE
    int eval_fen(const std::string&);
#else
    INLINE int eval_fen(const std::string&) { return 0; }
#endif
}
