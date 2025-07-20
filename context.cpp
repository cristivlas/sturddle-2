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
#include "common.h"
/*
 * Move ordering, board state evaluation, and other stuff
 * pertaining to the Context of the node being searched.
 */
#include <bitset>
#include <cerrno>
#include <chrono>
#include <iomanip>
#include <iterator>
#include <map>
#include <fstream>
#include <sstream>
#include "chess.h"

#define CONFIG_IMPL
  #include "context.h"
#undef CONFIG_IMPL

#if WITH_NNUE
  #include "nnue.h"
  #if !(SHARED_WEIGHTS)
    #include "weights.h"
  #endif
#endif


#include "eval.h"

using namespace chess;
using search::TranspositionTable;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

#if __linux__
#include <signal.h>

static void segv_handler(int sig)
{
    dump_backtrace(std::cerr);
    cython_wrapper::GIL_State gil_state;
    PyErr_SetString(PyExc_Exception, "Segmentation Fault");
}

static void setup_crash_handler()
{
    struct sigaction sa = {};
    sa.sa_handler = segv_handler;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGSEGV, &sa, nullptr) != 0)
    {
        perror("Failed to set signal handler");
        _exit(1); // Fail hard if we can't set up SIGSEGV handling
    }
}
#else

static void setup_crash_handler()
{
}
#endif /* __linux__ */

namespace
{
    class MovesCache
    {
        static constexpr size_t BUCKET_SIZE = 2;
        struct Entry
        {
            uint64_t    _hash = 0;
            MovesList   _moves;
            int         _use_count = 0;
            int         _write_attempts = 0;
        };

        std::vector<Entry> _data;

    public:
        explicit MovesCache(size_t size = 4000) : _data(size)
        {
            ASSERT_ALWAYS(size);
            ASSERT_ALWAYS(size % 2 == 0);
        }

        INLINE void clear()
        {
            std::fill_n(&_data[0], _data.size(), Entry());
        }

        INLINE bool lookup(const State& state, MovesList& moves)
        {
            const auto hash = state.hash();
            const auto slot = scramble64(hash);

            for (size_t j = 0; j < BUCKET_SIZE; ++j)
            {
                const auto i = (slot + j) & (_data.size() - 1);
                ASSERT(i < _data.size());
                auto& entry = _data[i];

                if (hash == entry._hash)
                {
                    ++entry._use_count;
                    moves.assign(entry._moves.begin(), entry._moves.end());
                    ASSERT(moves.size() == entry._moves.size());
                    return true;
                }
            }
            return false;
        }

        INLINE void write(const State& state, const MovesList& moves, bool force_write = false)
        {
            const auto hash = state.hash();
            const auto slot = scramble64(hash);

            for (size_t j = 0; j < BUCKET_SIZE; ++j)
            {
                const auto i = (slot + j) & (_data.size() - 1);
                ASSERT(i < _data.size());
                auto& entry = _data[i];

                if (force_write /* bypass eviction mechanism and forcefully write */
                    || hash == entry._hash
                    || ++entry._write_attempts > 2 * entry._use_count)
                {
                    entry._moves.assign(moves.begin(), moves.end());
                    ASSERT(entry._moves.size() == moves.size());
                    if (hash != entry._hash)
                        entry._use_count = 0;
                    entry._hash = hash;
                    entry._write_attempts = 0;
                    break;
                }
            }
        }
    };
} /* namespace */


/*
 * Late-move reduction tables (adapted from berserk)
 */
struct LMR
{
    int _table[PLY_MAX][64] = {};

    LMR()
    {
        for (int depth = 1; depth < PLY_MAX; ++depth)
        {
            for (int moves = 1; moves < 64; ++moves)
            {
                const auto v = 0.5 + log(depth) * log(moves) / 2;
                const auto e = (100 + depth) / 100.0;

                _table[depth][moves] = int(pow(v, e));
            }
        }
    }
} LMR;

/*---------------------------------------------------------------------------
 *
 * Configuration API, for tweaking parameters via Python scripts
 * (such as https://chess-tuning-tools.readthedocs.io/en/latest/)
 *
 *--------------------------------------------------------------------------*/
std::map<std::string, Param> _get_param_info()
{
    std::map<std::string, Param> info;

    for (const auto& elem : Config::_namespace)
    {
        info.emplace(elem.first,
            Param {
                *elem.second._val,
                elem.second._min,
                elem.second._max,
                elem.second._group,
                elem.second._normal
            });
    }

    info.emplace("Hash", Param {
        int(TranspositionTable::get_hash_size()),
        HASH_MIN,
        int(TranspositionTable::max_hash_size()),
        "Settings",
    });

    return info;
}


void _set_param(const std::string& name, int value, bool echo)
{
    if (name == "Hash")
    {
        const int HASH_MAX = int(TranspositionTable::max_hash_size());
        TranspositionTable::set_hash_size(std::max(std::min(value, HASH_MAX), HASH_MIN));

        if (echo)
            std::cout << "info string " << name << "=" << TranspositionTable::get_hash_size() << "\n";
        return;
    }

    const auto iter = Config::_namespace.find(name);

    if (iter == Config::_namespace.end())
    {
        search::Context::log_message(LogLevel::ERROR, "unknown parameter: \"" + name + "\"");
    }
    else if (value < iter->second._min || value > iter->second._max)
    {
        std::ostringstream err;
        err << name << ": " << value << " is out of range [";
        err << iter->second._min << ", " << iter->second._max << "]";

        search::Context::log_message(LogLevel::ERROR, err.str());
    }
    else
    {
        ASSERT_ALWAYS(iter->second._val);

        if (*iter->second._val != value)
        {
            *iter->second._val = value;

            if (echo)
                std::cout << "info string " << name << "=" << *iter->second._val << std::endl;
        }
    }
}


std::map<std::string, int> _get_params()
{
    std::map<std::string, int> cfg;
    for (const auto& elem : Config::_namespace)
    {
        cfg.emplace(elem.first, *elem.second._val);
    }
    cfg.emplace(std::string("Hash"), int(TranspositionTable::get_hash_size()));
    return cfg;
}


/*****************************************************************************
 *  NNUE
 *****************************************************************************/

#if WITH_NNUE

constexpr int INPUTS_A = 897;
constexpr int INPUTS_B = 256;
constexpr int HIDDEN_1A = 1280;
constexpr int HIDDEN_1A_POOLED = HIDDEN_1A / nnue::POOL_STRIDE;
constexpr int HIDDEN_1B = 64;
constexpr int HIDDEN_2 = 16;
constexpr int HIDDEN_3 = 16;

using LAttnType = nnue::Layer<HIDDEN_1B, 32>;
using L1AType = nnue::Layer<INPUTS_A, HIDDEN_1A, int16_t, nnue::QSCALE>;
using L1BType = nnue::Layer<INPUTS_B, HIDDEN_1B, int16_t, nnue::QSCALE>;
using L2Type = nnue::Layer<HIDDEN_1A_POOLED, HIDDEN_2>;
using L3Type = nnue::Layer<HIDDEN_2, HIDDEN_3>;
using EVALType = nnue::Layer<HIDDEN_3, 1>;
using LMOVEType = nnue::Layer<INPUTS_A, 4096, int16_t, nnue::QSCALE>;
/*
 * The accumulator takes the inputs and processes them into two outputs,
 * using layers L1A and L1B. L1B processes the 1st 256 inputs, which
 * correspond to kings and pawns. The output of L1B is processed by the
 * spatial attention layer, which moodulates the outputs of the L1A layer.
 */
using Accumulator = nnue::Accumulator<INPUTS_A, HIDDEN_1A, HIDDEN_1B>;
using AccumulatorStack = std::array<Accumulator, PLY_MAX>;

static std::vector<AccumulatorStack> NNUE_data(SMP_CORES);

#if !SHARED_WEIGHTS
static struct
{
    void init() {}

    LAttnType L_ATTN{spatial_attn_w, spatial_attn_b};
    L1AType L1A{hidden_1a_w, hidden_1a_b};
    L1BType L1B{hidden_1b_w, hidden_1b_b};
    L2Type L2{hidden_2_w, hidden_2_b};
    L3Type L3{hidden_3_w, hidden_3_b};
    EVALType EVAL{out_w, out_b};

#if USE_ROOT_MOVES
    LMOVEType L_M{move_w, move_b};
#endif /* USE_ROOT_MOVES */
} model;

#else
/* Weights are built as separate module and shared between all engine flavors. */
class WeightLoader
{
    PyObject* module = nullptr;

public:
    WeightLoader()
    {
        module = PyImport_ImportModule("weights");
        if (!module)
        {
            PyErr_Print();
            throw std::runtime_error("Failed to load weights module");
        }
    }

    ~WeightLoader()
    {
        if (module)
        {
            Py_DECREF(module);
            module = nullptr;
        }
    }

    template<typename T>
    T* get_weights(const char* name)
    {
        PyObject* func = PyObject_GetAttrString(module, name);
        ASSERT_MESSAGE(func, std::string(name));

        PyObject* capsule = PyObject_CallObject(func, nullptr);

        Py_DECREF(func);
        T* result = static_cast<T*>(PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule)));
        Py_DECREF(capsule);
        return result;
    }
};

template <typename T, size_t ROWS, size_t COLS>
class WeightAdapter
{
    const T* raw_ptr;

public:
    WeightAdapter(WeightLoader& loader, const char* name) : raw_ptr(loader.get_weights<T>(name)) {}

    const T (&as_2d() const)[ROWS][COLS] { return *reinterpret_cast<const T(*)[ROWS][COLS]>(raw_ptr); }
    const T (&as_1d() const)[COLS] { return *reinterpret_cast<const T(*)[COLS]>(raw_ptr); }
};

template <typename L>
INLINE void init_layer(WeightLoader& loader, L& layer, const char* get_w, const char* get_b)
{
    WeightAdapter<float, L::ROWS, L::COLS> w(loader, get_w);
    WeightAdapter<float, 1, L::COLS> b(loader, get_b);
    layer.set_weights(w.as_2d(), b.as_1d());
}

#define INIT_LAYER(layer, name) init_layer(loader, layer, "get_" #name "_w", "get_" #name "_b")

static struct
{
    LAttnType L_ATTN;
    L1AType L1A;
    L1BType L1B;
    L2Type L2;
    L3Type L3;
    EVALType EVAL;

#if USE_ROOT_MOVES
    LMOVEType L_M;
#endif /* USE_ROOT_MOVES */

    void init()
    {
        WeightLoader loader;

        INIT_LAYER(L_ATTN, spatial_attn);
        INIT_LAYER(L1A, hidden_1a);
        INIT_LAYER(L1B, hidden_1b);
        INIT_LAYER(L2, hidden_2);
        INIT_LAYER(L3, hidden_3);
        INIT_LAYER(EVAL, eval);

    #if USE_ROOT_MOVES
        INIT_LAYER(L_M, move);
    #endif /* USE_ROOT_MOVES */
    }
} model;
#endif /* SHARED_WEIGHTS */


score_t search::Context::eval_nnue_raw(bool update_only /* = false */, bool side_to_move_pov /* = true */)
{
    ASSERT(!is_valid(_eval_raw));
    const auto t = tid();

    auto& acc = NNUE_data[t][_ply];

    if (is_root())
    {
        acc.update(model.L1A, model.L1B, state());
    }
    else
    {
        auto& prev = NNUE_data[t][_ply - _nnue_prev_offs];

        if (prev.needs_update(_parent->state()))
        {
            _parent->eval_nnue_raw(true);
        }
        acc.update(model.L1A, model.L1B, _parent->state(), state(), _move, prev);
    }

    if (update_only)
    {
        _eval_raw = SCORE_MIN;
    }
    else
    {
        _eval_raw = nnue::eval(acc, model.L_ATTN, model.L2, model.L3, model.EVAL);

        if (side_to_move_pov)
        {
            _eval_raw *= SIGN[state().turn];
        }
    }

    return _eval_raw;
}


/* TODO: define array of margins, using LMP for now as a temporary hack. */

static INLINE score_t eval_margin(const Context& ctxt)
{
    const auto depth = ctxt.depth();
    const auto pc = ctxt.piece_count();

    return (NNUE_MAX_EVAL + search::LMP[depth]) * interpolate(pc, 100, 135) / 100.0;
}


void search::Context::eval_with_nnue()
{
    if (!is_valid(_eval))
    {
        if (is_valid(tt_entry()._eval))
        {
            ASSERT(!is_root());
            _eval = tt_entry()._eval;
            return;
        }

        auto eval = evaluate_material();

        if (state().just_king(!turn()) || (depth() >= 0 && abs(eval) <= eval_margin(*this)))
        {
        #if USE_ROOT_MOVES
            if (is_root() && iteration() == 1)
            {
                _move_maker.ensure_moves(*this, true);
            }
        #endif /* USE_ROOT_MOVES */

            const auto eval_nn = eval_nnue_raw();

        #if 0
            eval = eval_nn * (NNUE_EVAL_TERM + eval / 32) / 1024;
        #else
            eval = (eval_nn * NNUE_BLEND_PERCENT + eval * (100 - NNUE_BLEND_PERCENT)) / 100;
        #endif
        }
        else
        {
            /* Stick with material eval when heavily imbalanced, and assume NN */
            /* eval already accounts for insufficient material in branch above */

            eval = eval_insufficient_material(state(), eval,
                [eval](){
                    return eval;
                }
            );
        }

        _eval = eval + eval_fuzz();
    }
}


void search::Context::update_root_accumulators()
{
    const auto& root = NNUE_data[0][0];

    for (int i = 1; i != SMP_CORES; ++i)
    {
        auto& acc = NNUE_data[i][0];
        acc._hash = root._hash;
        memcpy(acc._output_a, root._output_a, sizeof(acc._output_a));
        memcpy(acc._output_b, root._output_b, sizeof(acc._output_b));

    #if DEBUG_INCREMENTAL
        memcpy(acc._input, root._input, sizeof(acc._input));
    #endif
    }
}

/* Test */
int nnue::eval_fen(const std::string& fen)
{
    auto ctxt = search::Context();
    chess::State state;
    ASSERT_ALWAYS(ctxt.tid() == 0);
    ASSERT_ALWAYS(ctxt._ply == 0);
    ctxt._state = &state;
    chess::parse_fen(fen, state);

    search::TranspositionTable tt;
    tt.init(true);
    ctxt.set_tt(&tt);

    return ctxt.eval_nnue_raw(false, false);
}
#endif /* WITH_NNUE */


namespace search
{
    /*---------------------------------------------------------------------
     * Context
     *---------------------------------------------------------------------*/
    atomic_bool Context::_cancel(false);
    atomic_int  Context::_tb_cardinality(6);
    atomic_int  Context::_time_limit(-1); /* milliseconds */
    atomic_time Context::_time_start;
    size_t Context::_callback_count(0);
    HistoryPtr Context::_history; /* moves played so far */
    std::vector<Context::ContextStack> Context::_context_stacks(SMP_CORES);
    std::vector<Context::MoveStack> Context::_move_stacks(SMP_CORES);
    std::vector<Context::StateStack> Context::_state_stacks(SMP_CORES);
    std::vector<MovesCache> _moves_cache(SMP_CORES);
    std::vector<PV> Context::_pvs(SMP_CORES);

    /* Cython callbacks */
    PyObject* Context::_engine = nullptr;

    bool (*Context::_book_init)(const std::string&) = nullptr;
    BaseMove (*Context::_book_lookup)(const State&, bool) = nullptr;
    std::string (*Context::_epd)(const State&) = nullptr;
    void (*Context::_log_message)(int, const std::string&, bool) = nullptr;

    void (*Context::_on_iter)(PyObject*, Context*, const IterationInfo*) = nullptr;
    void (*Context::_on_move)(PyObject*, const std::string&, int) = nullptr;
    void (*Context::_on_next)(PyObject*, int64_t) = nullptr;

    std::string(*Context::_pgn)(Context*) = nullptr;
    void (*Context::_print_state)(const State&, bool) = nullptr;
    void (*Context::_report)(PyObject*, std::vector<Context*>&) = nullptr;
    void (*Context::_set_syzygy_path)(const std::string&) = nullptr;
    bool (*Context::_tb_probe_wdl)(const State&, int*) = nullptr;
    size_t (*Context::_vmem_avail)() = nullptr;

    std::string Context::_syzygy_path = "syzygy/3-4-5";


    score_t eval_material_for_side_that_moved(const State& state, const State* prev, const BaseMove& move)
    {
        score_t eval;

        /* Get simple evaluation score from white's perspective */
        if (state.simple_score == State::UNKNOWN_SCORE)
        {
            if (prev && move)
                eval = state.eval_apply_delta(move, *prev);
            else
                eval = (state.simple_score = state.eval_simple());
        }
        else
        {
            ASSERT(state.simple_score == state.eval_simple());
            eval = state.simple_score;
        }


    #if EVAL_PIECE_GRADING

        /* eval_piece_grading applies adjustments from white's perspective */
        eval += eval_piece_grading(state, state.piece_count());

    #endif /* EVAL_PIECE_GRADING */

        /* Evaluate from the point of view of the side that just moved. */
        return eval * SIGN[!state.turn];
    }


    /* static */ void Context::clear_caches_and_stacks()
    {
        for (auto& cache : _moves_cache)
        {
            cache.clear();
        }

        for (auto& stack : _move_stacks)
        {
            for (auto& moves : stack)
                moves.clear();
        }

        for (auto& stack : _state_stacks)
        {
            for (auto& pool : stack)
                pool.clear();
        }
    }

    /* static */ void Context::init()
    {
        setup_crash_handler();
        _init(); /* Init attack masks and other magic bitboards in chess.cpp */

    #if WITH_NNUE
        model.init();
    #endif
    }


    /* Call into the Python logger */
    /* static */
    void Context::log_message(LogLevel level, const std::string& msg, bool force)
    {
        cython_wrapper::call(_log_message, int(level), msg, force);
    }


    /*
     * Copy context for two specific use cases:
     * 1) clone root at the beginning of SMP searches, and
     * 2) create a temporary context for singularity search.
     */
    Context* Context::clone(ContextBuffer& buffer, int ply) const
    {
        Context* ctxt = new (buffer.as_context(false)) Context;
        buffer._valid = true;

        ctxt->_algorithm = _algorithm;
        ctxt->_alpha = _alpha;
        ctxt->_beta = _beta;
        ctxt->_score = _score;
        ctxt->_eval = _eval;
        ctxt->_max_depth = _max_depth;
        ctxt->_parent = _parent;
        ctxt->_ply = ply;
        ctxt->_prev = _prev;
        ctxt->_state = &buffer._state;
        *ctxt->_state = this->state();
        ctxt->_move = _move;
        ctxt->_excluded = _excluded;
        ctxt->_counter_move = _counter_move;
        ctxt->_is_null_move = _is_null_move;
        ctxt->_double_ext = _double_ext;
        ctxt->_extension = _extension;

        std::copy_n(_path.begin(), _path_len, ctxt->_path.begin());
        ctxt->_path_len = _path_len;

        return ctxt;
    }


    /* Populate prev move from the Principal Variation, if missing. */
    void Context::ensure_prev_move()
    {
        if (!is_root() && !_prev && !is_null_move() && !_excluded)
        {
            const auto& pv = Context::_pvs[0];
            const size_t ply = _ply;

            if (ply + 1 < pv.size() && pv[ply] == _move)
            {
                _prev = pv[ply + 1];
            }
        }
    }


    /* static */ void Context::ensure_stacks()
    {
        const size_t n_threads(SMP_CORES);

        if (_context_stacks.size() < n_threads)
        {
            _context_stacks.resize(n_threads);
            _moves_cache.resize(n_threads);
            _move_stacks.resize(n_threads);
            _state_stacks.resize(n_threads);

            _pvs.resize(n_threads);
            for (size_t i = 0; i < n_threads; ++i)
                _pvs[i].reserve(PLY_MAX);

        #if WITH_NNUE
            NNUE_data.resize(n_threads);
        #endif
        }
    }


    /*
     * Track the best score and move so far, return true if beta cutoff;
     * called from search right after: score = -negamax(*next_ctxt).
     */
    bool Context::is_beta_cutoff(Context* next_ctxt, score_t score)
    {
        ASSERT(!next_ctxt->is_root());
        ASSERT(score > SCORE_MIN && score < SCORE_MAX);
        ASSERT(_alpha >= _score); /* invariant */

        if (score > _score)
        {
            if (next_ctxt->is_null_move())
            {
                /* consecutive null moves not allowed by design */
                ASSERT(!is_null_move());

                /* ignore if not fail-high */
                if (score < _beta)
                    return false;
            }

            if (score > _alpha)
            {
                if (!next_ctxt->is_null_move())
                {
                    if (next_ctxt->_prune_reason == PruneReason::PRUNE_TT
                        && next_ctxt->tt_entry()._depth >= depth() - 1)
                    {
                        /* Do not retry */
                        ASSERT(next_ctxt->tt_entry().is_valid());
                    }
                    else if (next_ctxt->_retry_above_alpha == RETRY::Reduced)
                    {
                        ASSERT(!next_ctxt->is_retry());
                        _retry_next = true;

                        if constexpr(EXTRA_STATS)
                            ++_tt->_retry_reductions;
                    }
                    else if (next_ctxt->_retry_above_alpha == RETRY::PVS && score < _beta)
                    {
                        ASSERT(!next_ctxt->is_retry());
                        _retry_next = true;
                        _retry_beta = -score;
                    }

                    if (_retry_next)
                    {
                        /* rewind and search again at full depth */
                        rewind(-1);
                        return false;
                    }

                    if (score >= _beta)
                    {
                        _cutoff_move = next_ctxt->_move;

                        if (is_null_move())
                        {
                            ASSERT(_parent);

                            if (next_ctxt->is_capture()) /* null move refuted by capture */
                                _parent->_capture_square = next_ctxt->_move.to_square();

                            if (score >= CHECKMATE - next_ctxt->depth())
                                _parent->_mate_detected = CHECKMATE - score + 1;
                        }
                    }
                }
                _alpha = score;
            }

            _score = score;

            if (!next_ctxt->is_null_move())
            {
                ASSERT(next_ctxt->_move._state == next_ctxt->_state);

                _best_move = next_ctxt->_move;

                _path_len = next_ctxt->_path_len;
                std::copy_n(next_ctxt->_path.begin(), _path_len, _path.begin());
            }
        }

        ASSERT(_alpha >= _score); /* invariant */

        return _alpha >= _beta;
    }


    /*
     * Called when there are no more moves available (endgame reached or
     * qsearch has examined all non-quiet moves in the current position).
     */
    score_t Context::evaluate_end()
    {
        /* precondition for calling this function */
        ASSERT(!has_moves());

        if (_pruned_count || _move_maker.have_skipped_moves())
        {
            ASSERT(!is_check());

            return evaluate<false>();
        }

        return is_check() ? checkmated(_ply) : 0;
    }


    /*
     * Make the capturing move, return false if not legal (verification optional).
     */
    static bool INLINE apply_capture(const State& state, State& next_state, const Move& move, bool verify = true)
    {
        state.clone_into(next_state);
        next_state.apply_move(move);

        ASSERT(next_state.turn != state.turn);
        ASSERT(next_state.is_capture());

        return !verify || !next_state.is_check(state.turn); /* legal move? */
    }


    template<bool Debug>
    int do_exchanges(const State& state, Bitboard mask, int tid, int ply)
    {
        ASSERT(popcount(mask) == 1); /* same square exchanges */
        ASSERT(ply >= PLY_MAX); /* use top half of moves stacks */

        ASSERT(state.simple_score != State::UNKNOWN_SCORE); /* incremental update */

        if (size_t(ply) >= Context::MAX_MOVE)
            return 0;

        auto& moves = Context::moves(tid, ply);
        state.generate_pseudo_legal_moves(moves, mask);

        /* sort moves by piece value */
        for (auto& move : moves)
        {
            ASSERT(state.piece_type_at(move.from_square()));
            move._score = state.piece_value_at(move.from_square(), state.turn);
        }
        /* sort lowest value attackers first */
        insertion_sort(moves.begin(), moves.end(),
            [](const Move& lhs, const Move& rhs) {
                return lhs._score < rhs._score;
            });

        if constexpr(Debug)
        {
            std::ostringstream out;
            out << "\tdo_exchanges (" << Context::epd(state) << ") ";
            for (const auto& move : moves)
                out << move << "(attacker=" << move._score << ") ";
            Context::log_message(LogLevel::DEBUG, out.str());
        }

        int score = 0;
        State next_state;

        /* iterate over pseudo-legal moves */
        for (const auto& move : moves)
        {
            ASSERT((BB_SQUARES[move.to_square()] & ~mask) == 0);
            ASSERT((state.kings & BB_SQUARES[move.to_square()]) == 0);

            apply_capture(state, next_state, move, false /* defer legality check */);

            const auto our_gain = capture_gain(state, next_state, move);

            if constexpr(Debug)
                Context::log_message(LogLevel::DEBUG, "\t>>> " + move.uci() + ": " + std::to_string(our_gain));

            ASSERT(our_gain > 0);
            if (our_gain <= score)
                break;

            if (next_state.is_check(state.turn))
                continue; /* not a legal move */

            if (ply + 1 >= PLY_MAX + EXCHANGES_MAX_DEPTH)
            {
                const auto their_best = estimate_static_exchanges(next_state, next_state.turn, move.to_square());
                score = std::max(score, our_gain - their_best);
            }
            else
            {
                next_state.castling_rights = 0;  /* castling moves do not capture */
                const auto their_best = do_exchanges<Debug>(next_state, mask, tid, ply + 1);

                if constexpr(Debug)
                {
                    std::ostringstream out;
                    out << "\t<<< " << move << ": " << our_gain << " - " << their_best;
                    Context::log_message(LogLevel::DEBUG, out.str());
                }

                score = std::max(score, our_gain - their_best);
            }
        }

        if constexpr(Debug)
            Context::log_message(LogLevel::DEBUG, "\tscore: " + std::to_string(score));

        return score;
    }


    /*
     * Look at all captures the side-to-move can make, "play" same square exchanges and return the best gain.
     *
     * Called by eval_captures (if !STATIC_EXCHANGES).
     */
    INLINE int
    do_captures(int tid, const State& state, Bitboard from_mask, Bitboard to_mask, score_t standpat_threshold)
    {
        ASSERT(!state.is_check(!state.turn)); /* expect legal position */

        static constexpr auto ply = FIRST_EXCHANGE_PLY;
        const auto mask = to_mask & state.occupied_co(!state.turn) & ~state.kings;

        /*
         * Generate all pseudo-legal captures.
         * NOTE: generate... function takes mask args in reverse: to, from.
         */
        auto& moves = Context::moves(tid, ply);
        if (state.generate_pseudo_legal_moves(moves, mask, from_mask).empty())
        {
            return 0;
        }

        bool standpat = true;

        /*
         * 1) Go over the captures and assign the victim value to each move score.
         */
        for (auto& move : moves)
        {
            ASSERT(state.piece_type_at(move.to_square()));
            ASSERT(state.piece_type_at(move.from_square()));
            ASSERT(state.piece_color_at(move.to_square()) != state.turn);

            move._score = state.piece_value_at(move.to_square(), !state.turn); /* victim value */
            if (const auto promo = move.promotion())
            {
                /* Take piece squares and piece grading (dynamic value) into account for the promo */
                const auto promo_val = state.piece_value_at(move.to_square(), state.turn, promo);
                ASSERT(USE_PIECE_SQUARE_TABLES || EVAL_PIECE_GRADING || WEIGHT[promo] == promo_val);

                move._score += promo_val - state.piece_value_at(move.from_square(), state.turn);
            }

            if (move._score + STANDPAT_MARGIN >= standpat_threshold)
                standpat = false;
        }

        if (standpat)
            return 0;

        /*
         * 2) Sort most valuable victim first.
         */
        insertion_sort(moves.begin(), moves.end(), [](const Move& lhs, const Move& rhs) {
            return lhs._score > rhs._score;
        });

        if constexpr(DEBUG_CAPTURES)
        {
            std::ostringstream out;
            out << "do_captures (" << Context::epd(state) << ") ";
            for (const auto& move : moves)
                out << move << "(" << move._score << ") ";

            Context::log_message(LogLevel::DEBUG, out.str());
        }

        int score = 0;
        State next_state;

        /*
         * 3) Go through the moves and "play out" the exchanges.
         */
        for (const auto& move : moves)
        {
            ASSERT(move._score > 0);

            /* potential gain is less than best score so far? bail */
            if (move._score /* + STANDPAT_MARGIN  */ <= score)
            {
                if constexpr(DEBUG_CAPTURES)
                {
                    std::ostringstream out;
                    out << "\t" << move << " " << move._score /* << " + " << STANDPAT_MARGIN */ << " <= " << score;
                    Context::log_message(LogLevel::DEBUG, out.str());
                }

                break;
            }

            if constexpr(DEBUG_CAPTURES)
                Context::log_message(LogLevel::DEBUG, "*** " + move.uci());

            if (!apply_capture(state, next_state, move))
                continue;

            const auto our_gain = capture_gain(state, next_state, move);

            ASSERT(USE_PIECE_SQUARE_TABLES || EVAL_PIECE_GRADING || our_gain > score);

            /****************************************************************/
            /* "play through" same square exchanges                         */
            next_state.castling_rights = 0; /* castling moves can't capture */
            const auto mask_to = BB_SQUARES[move.to_square()];
            const auto their_best = do_exchanges<DEBUG_CAPTURES>(next_state, mask_to, tid, ply + 1);

            const auto value = our_gain - their_best;

            /****************************************************************/
            if (value > score)
                score = value;

            if constexpr(DEBUG_CAPTURES)
            {
                std::ostringstream out;
                out << "\t" << move << ": " << value << " (" << our_gain << " - " << their_best << ") score: " << score;

                Context::log_message(LogLevel::DEBUG, out.str());
            }
        }

        return score;
    }


    score_t eval_captures(Context& ctxt, score_t score)
    {
        if constexpr(DEBUG_CAPTURES)
            ctxt.log_message(LogLevel::DEBUG, "eval_captures");

        const auto* const state = ctxt._state;

        score_t result;

        if constexpr(STATIC_EXCHANGES)
        {
            result = estimate_captures(*state);
        }
        else
        {
            const int standpat_threshold = ctxt._ply > 1 ? ctxt._alpha - score : SCORE_MIN;
            result = do_captures(ctxt.tid(), *state, BB_ALL, BB_ALL, standpat_threshold);
        }

        ASSERT(result >= 0);

        if constexpr(DEBUG_CAPTURES)
            ctxt.log_message(LogLevel::DEBUG, "captures: " + std::to_string(result));

        return result;
    }


    /*
     * Static evaluation has three components:
     * 1. base = material + piece-square table values (optional)
     * 2. tactical (positional) - aka Hand-crafted evals (HCE)
     * 3. capture estimates (in Context::evaluate)
     * NOTE: when using NNUE for evaluation (default), 2 does not apply.
     */
    score_t Context::_evaluate()
    {
        _tt->_eval_depth = std::max(_ply, _tt->_eval_depth);

        if (!is_valid(_eval))
        {
        #if WITH_NNUE
            eval_with_nnue();
        #else
            /*
             * 1. Material + piece-squares
             */
            _eval = state().eval();

            ASSERT(_eval > SCORE_MIN);
            ASSERT(_eval < SCORE_MAX);

            _eval += eval_fuzz();

            /*
             * 2. Tactical (positional) evaluation.
             */
            _eval += eval_insufficient_material(state(), _eval, [this]() {
                return eval_tactical(*this, _eval);
            });
        #endif /* !WITH_NNUE */
        }

        ASSERT(_eval > SCORE_MIN);
        ASSERT(_eval < SCORE_MAX);

        return _eval;
    }


    void Context::extend()
    {
    #if FRACTIONAL_EXTENSIONS
       /*
        * Fractional extensions: https://www.chessprogramming.org/Extensions
        * "[...] extension can be added that does not yet extend the search,
        * but further down the tree may cause an extension when another
        * fractional extension causes the net extension to exceed one ply."
        */
        if (_extension || depth() >= MIN_EXT_DEPTH)
        {
            /*
             * things that could add interestingness along the search path
             * "what is ultimately to be reduced must first be expanded" Lao Tzu
             */
            _extension += state().pushed_pawns_score;
            _extension += _move.from_square() == _parent->_capture_square;
            _extension += is_recapture() * (is_pv_node() * (ONE_PLY - 1) + 1);

            const auto double_extension_ok = (_double_ext <= DOUBLE_EXT_MAX);
            const auto extend = std::min(1 + double_extension_ok, _extension / ONE_PLY);

            ASSERT(extend >= 0);

            _max_depth += extend;
            _extension %= ONE_PLY;
            _double_ext += extend > 1;
        }
    #endif /* FRACTIONAL_EXTENSIONS */

        /* https://www.chessprogramming.org/Capture_Extensions */
        if (is_capture()
            && !is_extended()
            && abs(state().eval_material()) <= REBEL_EXTENSION_MARGIN
            && state().just_king_and_pawns()
            && !_parent->state().just_king_and_pawns())
        {
            _max_depth += REBEL_EXTENSION;
        }
    }


    /*
     * Used when the search has fails to find a move before the time runs out.
     */
    const Move* Context::first_valid_move()
    {
        rewind(0);
        return get_next_move(0 /* = no futility pruning */);
    }


    /*
     * Reinitialize top context at the start of a new iteration.
     */
    void Context::reinitialize()
    {
        ASSERT(is_root());
        ASSERT(!_is_null_move);
        ASSERT(_tt->_w_alpha <= _alpha);
        ASSERT(_retry_above_alpha == RETRY::None);
        ASSERT(_prune_reason == PruneReason::PRUNE_NONE);

        _best_move = BaseMove();
        _cancel = false;
        _can_forward_prune = -1;

        _capture_square = Square::UNDEFINED;
        _cutoff_move = Move();

        _extension = 0;
        _has_singleton = false;

        _max_depth = iteration();

        _mate_detected = 0;
        _multicut_allowed = MULTICUT;

        _null_move_allowed[WHITE] = true;
        _null_move_allowed[BLACK] = true;

        _prune_reason = PruneReason::PRUNE_NONE;

        _repetitions = -1;

        _retry_next = false;
        _retry_beta = SCORE_MAX;

        rewind(0, true);
    }


    static INLINE int window_delta(int iteration, int depth, double score)
    {
        return WINDOW_HALF * pow2(iteration) + WINDOW_COEFF * depth * log(1 + abs(score) / WINDOW_DIV);
    }


    void Context::set_search_window(score_t score, score_t& prev_score)
    {
        if (!ASPIRATION_WINDOW || iteration() <= 3)
        {
            _alpha = SCORE_MIN;
            _beta = SCORE_MAX;
        }
        else if (_mate_detected % 2)
        {
            _alpha = _tt->_w_alpha;
            _beta = SCORE_MAX;
        }
        else if (_mate_detected)
        {
            _alpha = SCORE_MIN;
            _beta = _tt->_w_beta;
        }
        else if (score <= _tt->_w_alpha)
        {
            _alpha = std::max<score_t>(SCORE_MIN, score - window_delta(iteration(), _tt->_eval_depth, score));
            _beta = _tt->_w_beta;
        }
        else if (score >= _tt->_w_beta)
        {
            _alpha = _tt->_w_alpha;
            _beta = std::min<score_t>(SCORE_MAX, score + window_delta(iteration(), _tt->_eval_depth, score));
        }
        else
        {
            const score_t delta = score - prev_score;
            prev_score = score;

        #if 1
            /* Widen in the OPPOSITE direction of the score trend, in case the raise/fall in
             * the score is caused by too narrow a window and a refutation / tactic was missed.
             */
            _alpha = std::max<score_t>(SCORE_MIN, score - std::max(WINDOW_HALF, delta));
            _beta = std::min<score_t>(SCORE_MAX, score + std::max(WINDOW_HALF, -delta));
        #else
            const score_t window_size = std::max(WINDOW_HALF, abs(delta));
            _alpha = std::max<score_t>(SCORE_MIN, score - window_size);
            _beta = std::min<score_t>(SCORE_MAX, score + window_size);
        #endif
        }

        /* save iteration bounds */
        _tt->_w_alpha = _alpha;
        _tt->_w_beta = _beta;

        _score = SCORE_MIN;
    }


    /*
     * Late move reduction and pruning.
     * https://www.chessprogramming.org/Late_Move_Reductions
     */
    LMRAction Context::late_move_reduce(int count)
    {
        ASSERT(!is_null_move());
        ASSERT(_parent);

        const int depth = this->depth();

        /* late move pruning */
        if (depth > 0 && count >= LMP[depth] && can_prune())
            return LMRAction::Prune;

        /* no reductions at very low depth and in qsearch */
        if (depth < 3 || count < LATE_MOVE_REDUCTION_THRESHOLD || !can_reduce())
            return LMRAction::None;

        /* Lookup reduction in the Late Move Reduction table. */
        auto reduction = LMR._table[std::min(depth, PLY_MAX-1)][std::min(count, 63)];

        if (_move._group != MoveOrder::TACTICAL_MOVES)
        {
            reduction += !_parent->has_improved();
            reduction -= 2 * _parent->is_counter_move(_move);

            if (get_tt()->_w_beta <= get_tt()->_w_alpha + 2 * WINDOW_HALF && iteration() >= 13)
                ++reduction;
        }

        if (is_capture() || (_move.from_square() == _parent->_capture_square))
            --reduction;

        const auto hist_score = _parent->history_score(_move);
        if (hist_score > 0 && hist_score < HISTORY_LOW)
            ++reduction;

        reduction = std::max(1, reduction);
        if (reduction > depth && can_prune())
            return LMRAction::Prune;

        ASSERT(reduction > 0);
        _max_depth -= reduction;

        /*
         * https://www.chessprogramming.org/Late_Move_Reductions
         * "Classical implementation assumes a re-search at full depth
         * if the reduced depth search returns a score above alpha."
         */
        if (!_retry_above_alpha)
            _retry_above_alpha = RETRY::Reduced;

        if constexpr(EXTRA_STATS)
            ++_tt->_reductions;

        return LMRAction::Ok;
    }


    bool Context::is_last_move()
    {
        return _parent && _parent->_move_maker.is_last(*_parent);
    }


    /*
     * Treat this node as a leaf or keep recursing down the graph?
     *
     * Once depth exceeds _max_depth only captures and some tactical
     * moves (such as pushed pawns) will be searched.
     */
    bool Context::is_leaf()
    {
        ASSERT(_fifty < 100);

        if (is_root())
            return false;
        else
            ASSERT(is_repeated() <= 0);

        if (_ply + 1 >= PLY_MAX)
            return true;

        /* the only available move? */
        if (_is_singleton && !get_tt()->_analysis)
        {
            ASSERT(_ply == 1);
            return true;
        }

        if (depth() > 0
            || is_null_move()
            || is_retry()
            || is_promotion()
            || is_check()
            /*
             * last move to search from current node, with score close to mate?
             * extend the search as to not miss a possible mate in the next move
             */
            || (_parent->_score < MATE_LOW && is_valid(_parent->_score) && is_last_move())
           )
            return false;

        /* treat it as leaf for now but retry and extend if it beats alpha */
        if (is_reduced() && !_retry_above_alpha)
            _retry_above_alpha = RETRY::Reduced;

        return true;
    }


    /*
     * https://en.wikipedia.org/wiki/Extended_Position_Description
     */
    std::string Context::epd(const State& state)
    {
        return cython_wrapper::call(_epd, state);
    }


    int64_t Context::check_time_and_update_nps()
    {
        const auto millisec = elapsed_milliseconds();

        /*
         * Update nodes-per-second for this thread.
         */
        if (millisec)
            _tt->set_nps((1000 * _tt->nodes()) / millisec);
        else
            _tt->set_nps(_tt->nodes());

        if (_time_limit >= 0 && millisec >= _time_limit)
        {
            cancel();
            return -1;
        }

        return millisec;
    }


    /*
     * Allow for apps to implement strength levels by slowing down the engine.
     */
    int64_t Context::nanosleep(int nanosec)
    {
#if defined(_POSIX_VERSION)
        timespec delay = { 0, nanosec };
        ::nanosleep(&delay, nullptr);
#else
        /* busy wait */
        const auto start = high_resolution_clock::now();

        while (true)
        {
            const auto now = high_resolution_clock::now();
            const auto count = duration_cast<nanoseconds>(now - start).count();

            if (count >= nanosec)
                break;

            if (check_time_and_update_nps() < 0)
                break;
        }
#endif /* _POSIX_VERSION */

        return check_time_and_update_nps();
    }


    void Context::cache_scores(bool force_write)
    {
        ASSERT(move_count() >= 0);
        auto& moves_list = moves();

        for (auto& move: moves_list)
        {
            move._old_score = move._score;
            move._old_group = move._group;
        }

        _moves_cache[tid()].write(state(), moves_list, force_write);
    }


    /*---------------------------------------------------------------------
     * MoveMaker
     *---------------------------------------------------------------------*/

    int MoveMaker::rewind(Context& ctxt, int where, bool force_reorder)
    {
        if (_count < 0)
            return -1;

        ASSERT(_count > 0 || where == 0);
        ASSERT(where == 0 || where == -1); /* other cases not supported */

        if (force_reorder)
        {
            ASSERT(!ctxt.is_retry());
            ASSERT(where == 0);

            _phase = 0;

            auto& moves_list = ctxt.moves();
            for (int i = 0; i != _count; ++i)
            {
                auto& move = moves_list[i];

                if (move._state == nullptr)
                {
                    ASSERT(move._group >= MoveOrder::UNORDERED_MOVES);
                    ASSERT(move._score == 0);
                    break;
                }

                if (move._group == MoveOrder::ILLEGAL_MOVES)
                {
                    _count = i;
                    break;
                }

                move._old_score = move._score;
                move._old_group = move._group;

                move._score = 0;
                move._group = MoveOrder::UNORDERED_MOVES;
            }

        }

        if (where >= 0)
        {
            _current = std::min(where, _count);
        }
        else
        {
            _current = std::max(0, _current + where);
        }

        ASSERT(_current >= 0);
        return _current;
    }


    void MoveMaker::generate_unordered_moves(Context& ctxt, bool order_root_moves)
    {
        /* pre-conditions */
        ASSERT(_count < 0);
        ASSERT(_current < 0);
        ASSERT(_phase == 0);
        ASSERT(_state_index == 0);

        bool from_cache = false;
        auto& moves_list = ctxt.moves();

        auto& moves_cache = _moves_cache[ctxt.tid()];
        if (moves_cache.lookup(ctxt.state(), moves_list))
        {
            from_cache = true;
            for (auto& move : moves_list)
            {
                move._state = nullptr;
                move._score = 0;
                move._group = MoveOrder::UNORDERED_MOVES;
            }
        }
        else
        {
            ctxt.state().generate_pseudo_legal_moves(moves_list);
        }

        _count = int(moves_list.size());
        _current = 0;

        auto& states_vec = Context::states(ctxt.tid(), ctxt._ply);
        if (states_vec.size() < size_t(_count))
        {
            states_vec.resize(_count);
        }
    #if GROUP_QUIET_MOVES
        /*
         * In quiescent search, only quiet moves are interesting.
         * If in check, no need to determine "quieteness", since
         * all legal moves are about getting out of check.
         */
        _group_quiet_moves = (ctxt.depth() < 0 && !ctxt.is_check());
    #endif /* GROUP_QUIET_MOVES */

    #if USE_ROOT_MOVES
        if (order_root_moves && ctxt._time_limit >= ROOT_MOVES_MIN_TIME)
        {
            int16_t input[nnue::round_up<16>(INPUTS_A)] = {};
            nnue::one_hot_encode(ctxt.state(), input);
            nnue::predict_moves(input, model.L_M, moves_list);

            bool all_valid = true;
            int count = 0;

            for (auto& move : moves_list)
            {
                ASSERT(move._group == MoveOrder::UNORDERED_MOVES);
                if (count >= ROOT_MAX_MOVES)
                {
                    move._score = 0;
                }
                else if (make_move<false>(ctxt, move, MoveOrder::ROOT_MOVES, move._score))
                {
                    // std::cout << "info string " << move << ": " << move._score / nnue::QSCALE << std::endl;
                    ++count;
                }
                else
                {
                    all_valid = move._score = 0;
                }
            }

            if (all_valid)
                _need_sort = false; /* moves already sorted by eval_with_moves */
            else
                sort_moves(ctxt, 0, moves_list.size());
        }
    #endif /* USE_ROOT_MOVES */

        if (!from_cache)
            moves_cache.write(ctxt.state(), moves_list);
    }


    static INLINE const Move* match_killer(const KillerMoves* killers, const Move& move)
    {
        if (killers)
            for (size_t i = 0; i != 2; ++i)
                if ((*killers)[i] == move)
                    return &(*killers)[i];

        return nullptr;
    }


    static INLINE bool is_pawn_push(Context& ctxt, const Move& move)
    {
        return ctxt.state().pawns & BB_PASSED[ctxt.turn()] & BB_SQUARES[move.from_square()];
    }


    template<std::size_t... I>
    static constexpr std::array<double, sizeof ... (I)> thresholds(std::index_sequence<I...>)
    {
        auto logistic = [](int i) { return HISTORY_HIGH / (1 + exp(6 - i)); };
        return { logistic(I) ... };
    }

    /* Phase 3 */
    static const auto hist_thresholds = thresholds(std::make_index_sequence<PLY_MAX>{});


    template<int Phase>
    INLINE void MoveMaker::order_moves_phase(
            Context&    ctxt,
            MovesList&  moves_list,
            size_t      start_at,
            size_t      count,
            score_t     futility)
    {
        const KillerMoves* const killer_moves = (Phase == 2 && ctxt.depth() > 0)
            ? ctxt._tt->get_killer_moves(ctxt._ply) : nullptr;

        /* Confidence bar for historical scores */
        const double hist_high = (Phase == 3) ? hist_thresholds[ctxt.iteration()] : 0;

        /********************************************************************/
        /* Iterate over pseudo-legal moves                                  */
        /********************************************************************/
        for (size_t i = start_at; i < count; ++i)
        {
            auto& move = moves_list[i];

            if (move._group >= MoveOrder::PRUNED_MOVES)
            {
                if (_need_sort)
                    continue;
                else
                    break;
            }

            ASSERT(move._group == MoveOrder::UNORDERED_MOVES);

            if constexpr (Phase == 1)  /* best moves from previous iteration and from hashtable */
            {
                if (move == ctxt._prev)
                {
                    make_move<false>(ctxt, move, ctxt._ply < 3 ? MoveOrder::PREV_ITER : MoveOrder::HASH_MOVES);
                }
                else if (move == ctxt.tt_entry()._best_move)
                {
                    make_move<false>(ctxt, move, MoveOrder::BEST_MOVES);
                }
                else if (move == ctxt.tt_entry()._hash_move)
                {
                    make_move<false>(ctxt, move, MoveOrder::HASH_MOVES, ctxt.tt_entry()._value);
                }
                else if (move.promotion())
                {
                    make_move<false>(ctxt, move, MoveOrder::PROMOTIONS, WEIGHT[move.promotion()]);
                }
                else if (((ctxt._move && move.to_square() == ctxt._move.to_square()) || ctxt.state().is_en_passant(move))
                    && make_move<false>(ctxt, move, MoveOrder::LAST_MOVED_CAPTURE))
                {
                    ASSERT(move._state->is_capture());
                    /*
                     * Looking at the capture of the last piece moved by the opponent before
                     * other captures may speed up the refutation of the current variation.
                     */

                    /* Sort in decreasing order of the capturing piece's value. */
                    move._score = -ctxt.state().piece_value_at(move.from_square(), ctxt.turn());
                }
            }
            /* Captures and killer moves. */
            else if constexpr (Phase == 2)
            {
                if (move._state ? move._state->is_capture() : ctxt.state().is_capture(move))
                {
                    make_capture(ctxt, move);
                    ASSERT(move._group != MoveOrder::UNORDERED_MOVES);
                }
                else if (auto k_move = match_killer(killer_moves, move))
                {
                    if (make_move<false>(ctxt, move, MoveOrder::KILLER_MOVES, k_move->_score))
                    {
                        if constexpr(EXTRA_STATS)
                            ++ctxt.get_tt()->_killers;
                    }
                }
            }
            /* Top historical scores, including counter-move bonus. */
            else if constexpr (Phase == 3)
            {
                const auto hist_score = ctxt.history_score(move);

                if (hist_score > hist_high)
                {
                    make_move<true>(ctxt, move, MoveOrder::HISTORY_COUNTERS, hist_score);
                }
                else if (move._old_group == MoveOrder::TACTICAL_MOVES)
                {
                    remake_move(ctxt, move);
                }
                else if (ctxt.is_counter_move(move)
                    || move.from_square() == ctxt._capture_square
                    || is_pawn_push(ctxt, move))
                {
                    if (make_move<true>(ctxt, move, MoveOrder::TACTICAL_MOVES, hist_score))
                        ASSERT(move._score == hist_score);
                }
            }
            else /* Phase == 4 */
            {
                if (move._old_group == MoveOrder::LATE_MOVES)
                {
                    remake_move(ctxt, move);
                    continue;
                }
                if (make_move<true>(ctxt, move, futility))
                {
                    incremental_update(move, ctxt);
                    const auto eval = eval_material_for_side_that_moved(*move._state, ctxt._state, move);
                    move._group = MoveOrder::LATE_MOVES;
                    move._score = ctxt.history_score(move) / (1 + HISTORY_LOW) + eval;
                }
            }
        }
    }


    /*
     * Order moves in multiple phases (passes). The idea is to minimize make_move() calls,
     * which validate the legality of the move. The granularity (number of phases) should
     * not be too high, to keep the overhead of sorting the moves low.
     */
    void MoveMaker::order_moves(Context& ctxt, size_t start_at, score_t futility)
    {
        ASSERT(_phase <= MAX_PHASE);
        ASSERT(ctxt._tt);
        ASSERT(ctxt.moves()[start_at]._group == MoveOrder::UNORDERED_MOVES);

        _have_move = false;

        auto& moves_list = ctxt.moves();
        const auto count = size_t(_count);
        ASSERT(count <= moves_list.size());

        while (!_have_move && start_at < count && _phase++ < MAX_PHASE)
        {
            switch (_phase)
            {
            case 1: order_moves_phase<1>(ctxt, moves_list, start_at, count, futility); break;
            case 2: order_moves_phase<2>(ctxt, moves_list, start_at, count, futility); break;
            case 3: order_moves_phase<3>(ctxt, moves_list, start_at, count, futility); break;
            case 4: order_moves_phase<4>(ctxt, moves_list, start_at, count, futility); break;
            default: ASSERT(false); break;
            }
        }

        ASSERT(_count >= 0);

        if (size_t(_count) <= start_at)
        {
            ASSERT(moves_list[start_at]._group >= MoveOrder::PRUNED_MOVES);
            _need_sort = false;
        }
        else if (_need_sort)
        {
            sort_moves(ctxt, start_at, count);
        }

    #if !defined(NO_ASSERT)
        for (size_t i = 1; i < moves_list.size(); ++i)
        {
            ASSERT(compare_moves_ge(moves_list[i-1], moves_list[i]));
        }
    #endif /* NO_ASSERT */
    }


    score_t Context::eval(bool as_white, int depth, int millisec)
    {
        if (millisec > 0)
            set_time_limit_ms(millisec);

        const auto score = search::iterative(*this, *get_tt(), depth + 1);
        return as_white ? score * SIGN[turn()] : score;
    }
} /* namespace */


/* Test */
score_t eval(const std::string& fen, bool as_white, int depth, int millis)
{
    auto ctxt = search::Context();
    ASSERT(ctxt.tid() == 0);
    ASSERT(ctxt._ply == 0);

    chess::State state;
    ctxt._state = &state;

    chess::parse_fen(fen, state);
    ASSERT(state.piece_count() == chess::popcount(state.occupied()));

    search::TranspositionTable tt;
    tt.init(true);
    ctxt.set_tt(&tt);

    return ctxt.eval(as_white, depth, millis);
}


void cancel_search(CancelReason reason)
{
    search::Context::cancel();

    switch (reason)
    {
    case CancelReason::PY_SIGNAL:
        std::cout << "\ninterrupted\n";
        _exit(1);

    case CancelReason::PY_ERROR:
        PyErr_Print();
        _exit(2);
    }
}
