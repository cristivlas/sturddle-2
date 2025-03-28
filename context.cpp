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
#include <cerrno>
#include <chrono>
#include <iomanip>
#include <iterator>
#include <map>
#include <fstream>
#include <sstream>
#include "chess.h"
#include "weights.h"
#include "nlohmann/json.hpp"

#define CONFIG_IMPL
  #include "context.h"
#undef CONFIG_IMPL

#if USE_VECTOR
  #include <xmmintrin.h>
#endif

#if !defined(WITH_NNUE)
  #define WITH_NNUE false
#endif

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
            State       _state;
            MovesList   _moves;
            int         _use_count = 0;
            int         _write_attempts = 0;
        };

        std::vector<Entry> _data;

    public:
        explicit MovesCache(size_t size = 4007) : _data(size)
        {
        }

        INLINE void clear()
        {
            std::vector<Entry>(_data.size()).swap(_data);
        }

        INLINE bool lookup(const State& state, MovesList& moves)
        {
            const auto hash = state.hash();
            const auto slot = scramble64(hash);

            for (size_t j = 0; j < BUCKET_SIZE; ++j)
            {
                const auto i = (slot + j) % _data.size();
                auto& entry = _data[i];
                if (hash == entry._state.hash() && state == entry._state)
                {
                    ++entry._use_count;
                    moves.assign(entry._moves.begin(), entry._moves.end());

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
                const auto i = (slot + j) % _data.size();
                auto& entry = _data[i];

                if (force_write /* bypass eviction mechanism and forcefully write */
                    || hash == entry._state.hash()
                    || ++entry._write_attempts > 2 * entry._use_count)
                {
                    entry._moves.assign(moves.begin(), moves.end());
                    if (hash != entry._state.hash())
                        entry._use_count = 0;
                    entry._state = state;
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
    int _table[PLY_MAX][64] = { { 0 }, { 0 } };

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
        if (WITH_NNUE && elem.second._group == "Eval")
            continue;

        info.emplace(elem.first,
            Param {
                *elem.second._val,
                elem.second._min,
                elem.second._max,
                elem.second._group
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
    else if (WITH_NNUE && iter->second._group == "Eval" && name.find("MOBILITY") != 0)
    {
        search::Context::log_message(LogLevel::WARN, "not used in NNUE mode: \"" + name + "\"");
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


void assert_param_ref()
{
#if REFCOUNT_PARAM
    for (auto& p : Config::_namespace)
    {
        if (p.second._val->_refcount == 0)
            search::Context::log_message(LogLevel::ERROR, p.first + ": unreferenced");

        ASSERT_ALWAYS(p.second._val->_refcount);
        p.second._val->_refcount = 0;
    }
#endif /* REFCOUNT_PARAM */
}


/*****************************************************************************
 *  NNUE
 *****************************************************************************/

#if WITH_NNUE

#ifndef _countof
#define _countof(x) std::extent<decltype(x)>::value
#endif

constexpr int INPUTS_A = 897;
constexpr int INPUTS_B = 256;
constexpr int HIDDEN_1A = 640;
constexpr int HIDDEN_1A_POOLED = HIDDEN_1A / 4;
constexpr int HIDDEN_1B = 64;
constexpr int HIDDEN_2 = 16;
constexpr int HIDDEN_3 = 16;

using Accumulator = nnue::Accumulator<INPUTS_A, HIDDEN_1A, HIDDEN_1B>;

static std::vector<std::array<Accumulator, PLY_MAX>> NNUE_data(SMP_CORES);

/*
 * The accumulator takes the inputs and processes them into two outputs,
 * using (hidden) neural layers L1A and L1B. L1B processes only the 1st
 * 128 inputs, which correspond to kings and pawns. The output of L1B is
 * processed by the dynamic weights layer (attention layer). The outputs
 * of the dynamic weights (attention) layer are multiplied element-wise
 * with the result of the L1A layer.
 */
static nnue::Layer<INPUTS_A, HIDDEN_1A, int16_t, nnue::QSCALE> L1A(hidden_1a_w, hidden_1a_b);
static nnue::Layer<INPUTS_B, HIDDEN_1B, int16_t, nnue::QSCALE> L1B(hidden_1b_w, hidden_1b_b);
static nnue::Layer<HIDDEN_1A_POOLED, HIDDEN_2> L2(hidden_2_w, hidden_2_b);
static nnue::Layer<HIDDEN_1B, _countof(dynamic_weights_b)> L_DYN(dynamic_weights_w, dynamic_weights_b);

static nnue::Layer<HIDDEN_2, HIDDEN_3> L3(hidden_3_w, hidden_3_b);
static nnue::Layer<HIDDEN_3, 1> L4(out_w, out_b);

using WeightSetter = std::function<void(const std::vector<std::vector<float>>&, const std::vector<float>&)>;
static std::unordered_map<std::string, WeightSetter> registry = {
    { "hidden_1a", [](const std::vector<std::vector<float>>& w, const std::vector<float>& b) { L1A.set_weights(w, b); } },
    { "hidden_1b", [](const std::vector<std::vector<float>>& w, const std::vector<float>& b) { L1B.set_weights(w, b); } },
    { "hidden_2", [](const std::vector<std::vector<float>>& w, const std::vector<float>& b) { L2.set_weights(w, b); } },
    { "hidden_3", [](const std::vector<std::vector<float>>& w, const std::vector<float>& b) { L2.set_weights(w, b); } },
    { "dynamic_weights", [](const std::vector<std::vector<float>>& w, const std::vector<float>& b) { L_DYN.set_weights(w, b); } },
    { "out", [](const std::vector<std::vector<float>>& w, const std::vector<float>& b) { L4.set_weights(w, b); } },
};


score_t search::Context::eval_nnue_raw(bool update_only /* = false */, bool side_to_move_pov /* = true */)
{
    ASSERT(!is_valid(_eval_raw));
    const auto t = tid();

    auto& acc = NNUE_data[t][_ply];

    if (is_root() || _non_incremental_update || _ply > PLY_MAX / 2)
    {
        acc.update(L1A, L1B, state());
    }
    else
    {
        auto& prev = NNUE_data[t][_ply - 1];

        if (prev.needs_update(_parent->state()))
        {
            _parent->eval_nnue_raw(true);
        }
        acc.update(L1A, L1B, _parent->state(), state(), _move, prev);
    }

    if (update_only)
    {
        _eval_raw = SCORE_MIN;
    }
    else
    {
        _eval_raw = nnue::eval(acc, L_DYN, L2, L3, L4);

        if (side_to_move_pov)
        {
            _eval_raw *= SIGN[state().turn];
        }

    #if DATAGEN
        /* Make sure that insufficient material conditions are detected. */

        _eval_raw = eval_insufficient_material(state(), _eval_raw, [this](){ return _eval_raw; });
    #endif
    }
    return _eval_raw;
}


void search::Context::eval_nnue()
{
    if (!is_valid(_eval))
    {
        if (is_valid(_tt_entry._eval))
        {
            _eval = _tt_entry._eval;
            return;
        }

        auto eval = evaluate_material();

        /* Stick with material eval when heavily imbalanced */
        /* TODO: define array of margins, using LMP for now as a temporary hack. */

        if (state().just_king(!turn())
            || (!is_leaf_extended() && abs(eval) <= NNUE_MAX_EVAL + LMP[depth()]))
        {
            /* NOTE: assume NNUE eval already accounts for insufficient material */
            eval = eval_nnue_raw() * (NNUE_EVAL_TERM + eval / 32) / 1024;
        }
        else
        {
            eval = eval_insufficient_material(state(), eval, [eval](){ return eval; });
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


static void set_default_model()
{
    L1A.set_weights(hidden_1a_w, hidden_1a_b);
    L1B.set_weights(hidden_1b_w, hidden_1b_b);

    L2.set_weights(hidden_2_w, hidden_2_b);
    L_DYN.set_weights(dynamic_weights_w, dynamic_weights_b);
    L4.set_weights(out_w, out_b);
}


static void load_model(const std::string& json_file_path)
{
    std::ifstream file(json_file_path);
    nlohmann::json weights_json;
    file >> weights_json;

    for (auto& element : weights_json.items())
    {
        std::string layer_name = element.key();
        auto weights_and_biases = element.value();

        // TODO: validate
        // const int input_dim = weights_and_biases["input_dim"];
        // const int output_dim = weights_and_biases["output_dim"];

        const auto weights = weights_and_biases["weights"].get<std::vector<std::vector<float>>>();
        const auto biases = weights_and_biases["biases"].get<std::vector<float>>();

        auto it = registry.find(layer_name);
        if (it == registry.end())
            throw std::runtime_error("no such layer: " + layer_name);
        else
            it->second(weights, biases);

        search::Context::log_message(LogLevel::INFO, json_file_path + ": " + layer_name);
    }

}


/*
 * Load neural net model from JSON.
 *
 * (tools/nnue/modeltojson saves TensorFlow model params as JSON)
 */
void search::Context::load_nnue_model(const std::string& json_file_path)
{
    try
    {
        load_model(json_file_path);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        log_message(LogLevel::ERROR, e.what());

        set_default_model();
    }

    /* reset accumulators */
    for (int i = 0; i != SMP_CORES; ++i)
        for (size_t j = 0; j != NNUE_data[i].size(); ++j)
            NNUE_data[i][j]._hash = 0;
}


int nnue::eval_fen(const std::string& fen)
{
    auto ctxt = search::Context();
    chess::State state;
    ASSERT_ALWAYS(ctxt.tid() == 0);
    ASSERT_ALWAYS(ctxt._ply == 0);
    ctxt._state = &state;
    chess::parse_fen(fen, state);
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


    /* static */ void Context::clear_moves_cache()
    {
        for (auto& cache : _moves_cache)
        {
            cache.clear();
        }
    }

    /* Init attack masks and other magic bitboards in chess.cpp */
    /* static */ void Context::init()
    {
        setup_crash_handler();
        _init();
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
        Context* ctxt = new (buffer.as_context()) Context;

        ctxt->_algorithm = _algorithm;
        ctxt->_alpha = _alpha;
        ctxt->_beta = _beta;
        ctxt->_score = _score;
        ctxt->_max_depth = _max_depth;
        ctxt->_parent = _parent;
        ctxt->_ply = ply;
        ctxt->_prev = _prev;
        ctxt->_state = &buffer._state;
        *ctxt->_state = this->state();
        ctxt->_move = _move;
        ctxt->_excluded = _excluded;
        ctxt->_tt_entry = _tt_entry;
        ctxt->_counter_move = _counter_move;
        ctxt->_is_null_move = _is_null_move;
        ctxt->_double_ext = _double_ext;
        ctxt->_extension = _extension;
        return ctxt;
    }


    /*
     * Lookup move in the principal variation from the previous iteration.
     * https://www.chessprogramming.org/PV-Move
     */
    static INLINE const BaseMove* lookup_pv(const Context& ctxt)
    {
        ASSERT(ctxt.get_tt());

        const auto& pv = ctxt.get_tt()->get_pv();
        const size_t ply = ctxt._ply;

        if (ply + 1 >= pv.size())
            return nullptr;

        return (pv[ply] == ctxt._move) ? &pv[ply + 1] : nullptr;
    }


    /* Populate prev move from the Principal Variation, if missing. */
    void Context::ensure_prev_move()
    {
        if (!is_root() && !_prev && !is_null_move() && !_excluded)
        {
            if (const auto move = lookup_pv(*this))
            {
                _prev = *move;
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
                        && next_ctxt->_tt_entry._depth >= depth() - 1)
                    {
                        /* Do not retry */
                        ASSERT(next_ctxt->_tt_entry.is_valid());
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

            return evaluate();
        }

        return is_check() ? checkmated(_ply) : 0;
    }


    /*
     * Make the capturing move, return false if not legal.
     */
    static bool INLINE apply_capture(const State& state, State& next_state, const Move& move)
    {
        state.clone_into(next_state);
        next_state.apply_move(move);

        ASSERT(next_state.turn != state.turn);
        ASSERT(next_state.capture_value > 0);

        return !next_state.is_check(state.turn); /* legal move? */
    }


    template<bool Debug>
    int do_exchanges(const State& state, Bitboard mask, score_t gain, int tid, int ply)
    {
        ASSERT(popcount(mask) == 1);
        ASSERT(gain >= 0);

        /* use top half of moves stacks */
        ASSERT(ply >= PLY_MAX);

        if (size_t(ply) >= Context::MAX_MOVE)
            return 0;

        mask &= ~state.kings;

        auto& moves = Context::moves(tid, ply);
        state.generate_pseudo_legal_moves(moves, mask);

        /* sort moves by piece type */
        for (auto& move : moves)
        {
            ASSERT(state.piece_type_at(move.from_square()));
            move._score = state.piece_weight_at(move.from_square());
        }
        /* sort lower-value attackers first */
        insertion_sort(moves.begin(), moves.end(),
            [](const Move& lhs, const Move& rhs) {
                return lhs._score < rhs._score;
            });

        if constexpr(Debug)
        {
            std::ostringstream out;
            out << "\tdo_exchanges (" << Context::epd(state) << ") gain=" << gain << " ";
            for (const auto& move : moves)
                out << move.uci() << "(" << move._score << ") ";
            Context::log_message(LogLevel::DEBUG, out.str());
        }

        int score = 0;
        int moves_count = 0;

        (void) moves_count; /* silence off compiler warning */

        State next_state;

        /* iterate over pseudo-legal moves */
        for (const auto& move : moves)
        {
            ASSERT((BB_SQUARES[move.to_square()] & ~mask) == 0);
            ASSERT((state.kings & BB_SQUARES[move.to_square()]) == 0);

            if (!apply_capture(state, next_state, move))
                continue;

            ++moves_count;

            if constexpr(Debug)
                Context::log_message(LogLevel::DEBUG, "\t>>> " + move.uci());

            const score_t capturer_value = move._score;
            ASSERT(capturer_value == state.piece_weight_at(move.from_square()));

            /*
             * If the value of the capture exceeds the other's side gain plus the value of the
             * capturing piece there is no need to call ourselves recursively, as even in the
             * worst case scenario of the capturer being taken the difference cannot be offset.
             */
            if (next_state.capture_value > gain + capturer_value)
            {
            #if EXCHANGES_DETECT_CHECKMATE
                if (next_state.is_checkmate())
                {
                    score = CHECKMATE - (ply + 1 - FIRST_EXCHANGE_PLY);

                    if constexpr(Debug)
                    {
                        std::ostringstream out;
                        out << "\t<<< " << move.uci() << ": CHECKMATE " << score;
                        Context::log_message(LogLevel::DEBUG, out.str());
                    }

                    break; /* impractical to keep looping in hope of a faster mate */
                }
            #endif /* EXCHANGES_DETECT_CHECKMATE */

                score = std::max(score, next_state.capture_value - gain - capturer_value);

                if constexpr(Debug)
                {
                    std::ostringstream out;
                    out << "\t<<< " << move.uci() << ": " << next_state.capture_value
                        << " - " << gain << " - " << capturer_value;
                    Context::log_message(LogLevel::DEBUG, out.str());
                }

                break; // moves are sorted by piece weight
            }

            /****************************************************************/
            score_t other = 0;

            /* Call recursively if the capture offsets the opponent's gain. */
            if (next_state.capture_value >= gain)
            {
                next_state.castling_rights = 0;  /* castling do not capture */

                other = do_exchanges<Debug>(
                    next_state,
                    mask,
                    next_state.capture_value - gain,
                    tid,
                    ply + 1);
            }
            /*****************************************************************/
            if constexpr(Debug)
            {
                std::ostringstream out;
                out << "\t<<< " << move.uci() << ": "
                    << next_state.capture_value << " - " << other;
                Context::log_message(LogLevel::DEBUG, out.str());
            }

            /* could continue and look for a quicker mate, but impractical */
            if (other < MATE_LOW)
                return -other;

            score = std::max(score, next_state.capture_value - other);
        }

    #if EXCHANGES_DETECT_CHECKMATE
        if (moves_count == 0 && state.is_checkmate())
        {
            score = -CHECKMATE + ply - FIRST_EXCHANGE_PLY;
        }
    #endif /* EXCHANGES_DETECT_CHECKMATE */

        if constexpr(Debug)
        {
            std::ostringstream out;
            out << "\tscore: " << score;
            Context::log_message(LogLevel::DEBUG, out.str());
        }

        return score;
    }


    /*
     * Look at all captures the side-to-move can make, "play through"
     * same square exchanges and return lower bound of maximum gain.
     *
     * Skip the exchanges when the value of the captured piece exceeds
     * the value of the capturer.
     *
     * Called by eval_captures (if !STATIC_EXCHANGES).
     */
    INLINE int do_captures(int tid, const State& state, Bitboard from_mask, Bitboard to_mask, score_t standpat_threshold)
    {
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
         * 1) Go over the captures and assign the "victim" value to each move.
         */
        for (auto& move : moves)
        {
            ASSERT(state.piece_type_at(move.to_square()));
            ASSERT(state.piece_type_at(move.from_square()));

            move._score = state.piece_weight_at(move.to_square());

            if (move._score + STANDPAT_MARGIN >= standpat_threshold)
                standpat = false;

            move._score -= state.piece_weight_at(move.from_square());
        }

        if (standpat)
            return 0;

        /*
         * 2) Sort most valuable victims, least valuable attacker first.
         */
        insertion_sort(moves.begin(), moves.end(), [](const Move& lhs, const Move& rhs) {
            return lhs._score > rhs._score;
        });

        if constexpr(DEBUG_CAPTURES)
        {
            std::ostringstream out;
            out << "do_captures (" << Context::epd(state) << ") ";
            for (const auto& move : moves)
                out << move.uci() << "(" << move._score << ") ";

            Context::log_message(LogLevel::DEBUG, out.str());
        }

        int score = 0;
        State next_state;

        /*
         * 3) Go through the moves and "play out" the exchanges.
         */
        for (const auto& move : moves)
        {
            /* do not expect to capture the king */
            ASSERT((state.kings & BB_SQUARES[move.to_square()]) == 0);

        #if !EXCHANGES_DETECT_CHECKMATE
            /* victim values less than what we got so far? bail */
            if (move._score <= score)
            {
                if constexpr(DEBUG_CAPTURES)
                {
                    std::ostringstream out;
                    out << "\t" << move.uci() << " " << move._score << " <= " << score;
                    Context::log_message(LogLevel::DEBUG, out.str());
                }

                break;
            }
        #endif /* !EXCHANGES_DETECT_CHECKMATE */

            if constexpr(DEBUG_CAPTURES)
                Context::log_message(LogLevel::DEBUG, "*** " + move.uci());

            /****************************************************************/
            if (!apply_capture(state, next_state, move))
                continue;

            ASSERT(next_state.capture_value > score || EXCHANGES_DETECT_CHECKMATE);
            ASSERT(move._score == next_state.capture_value - state.piece_weight_at(move.from_square()));

            const auto gain = move._score;

            /*
             * Worst case scenario the attacker gets captured, capturing
             * side still has a nice gain; skip "playing" the exchanges.
             */
            if (gain > 0)
            {
            #if !EXCHANGES_DETECT_CHECKMATE
                return gain;
            #else
                if (next_state.is_checkmate())
                    return CHECKMATE - (ply + 1 - FIRST_EXCHANGE_PLY);
                if constexpr(DEBUG_CAPTURES)
                    Context::log_message(
                        LogLevel::DEBUG,
                        move.uci() + ": skip exchanges: " + std::to_string(gain));
                if (gain > score)
                    score = gain;
                continue;
            #endif /* EXCHANGES_DETECT_CHECKMATE */
            }

            /****************************************************************/
            /* "play through" same square exchanges                         */
            next_state.castling_rights = 0; /* castling moves can't capture */

            const auto other = do_exchanges<DEBUG_CAPTURES>(
                next_state,
                BB_SQUARES[move.to_square()],
                next_state.capture_value,
                tid,
                ply + 1);
            /****************************************************************/

            if (other < MATE_LOW)
            {
                if constexpr(DEBUG_CAPTURES)
                    Context::log_message(LogLevel::DEBUG, move.uci() + ": checkmate");

                return -other;
            }
            const auto value = next_state.capture_value - other;

            if (value > score)
                score = value;

            if constexpr(DEBUG_CAPTURES)
            {
                std::ostringstream out;
                out << "\t" << move.uci() << ": " << value << " ("
                    << next_state.capture_value << " - " << other
                    << ") score: " << score;

                Context::log_message(LogLevel::DEBUG, out.str());
            }
        }
        return score;
    }


    score_t eval_captures(Context& ctxt, score_t score)
    {
        if (is_valid(ctxt._tt_entry._captures) && ctxt._tt_entry._depth >= ctxt.depth())
            return ctxt._tt_entry._captures;

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

        ctxt._tt_entry._captures = result;
        return result;
    }

#if !WITH_NNUE
    /*----------------------------------------------------------------------
     * Tactical evaluations.
     * Hand-crafted evaluations are not compiled when using NNUE evals.
     * All tactical scores are computed from the white side's perspective.
     *----------------------------------------------------------------------*/
    static INLINE int eval_center(const State& state, int pc)
    {
        int attacks = 0;
        int occupancy = 0;

        for (auto color : { BLACK, WHITE })
        {
            const auto s = SIGN[color];
            occupancy += s * popcount(state.pawns & state.occupied_co(color) & BB_CENTER);

            for_each_square(BB_CENTER, [&](Square square) {
                attacks += s * popcount(state.pawns & BB_PAWN_ATTACKS[!color][square]);
            });

        }
        return attacks * interpolate(pc, CENTER_ATTACKS, 0)
            + occupancy * interpolate(pc, CENTER_OCCUPANCY, 0);
    }


    static INLINE Bitboard king_area(const State& state, Square king)
    {
        const auto f = square_file(king);
        const auto r = square_rank(king);

        const auto file = BB_FILES[f + (f == 0) - (f == 7)];
        const auto rank = BB_RANKS[r + (r == 0) - (r == 7)];

        return(file | shift_left(file) | shift_right(file))
            & (rank | shift_up(rank) | shift_down(rank))
            & ~BB_SQUARES[king];
    }


    /*
     * Count friendly pieces minus opponent's pieces in the king
     * quadrant, as an approximate measure of the king's safety.
     */
    static INLINE int eval_king_quadrant(const State& state, int pcs)
    {
        int score = 0;

        for (auto color: {BLACK, WHITE})
        {
            const auto ours = state.occupied_co(color);
            const auto theirs = state.occupied_co(!color);
            const auto king_mask = state.kings & ours;

            for (auto q : BB_QUANDRANTS)
            {
                if (king_mask & q)
                {
                    q &= ~king_mask;

                    score += SIGN[color] * (
                        popcount(q & ours) - popcount(q & theirs)
                        /* count queens as 2 pieces */
                        + popcount(q & state.queens & ours)
                        - popcount(q & state.queens & theirs));

                    break;
                }
            }
        }

        return score * interpolate(pcs, MIDGAME_KING_QUADRANT, ENDGAME_KING_QUADRANT);
    }


    static INLINE int eval_king_safety(const State& state, int pcs)
    {
        int attacks = 0; /* attacks around the king */
        int castle  = 0; /* castling rights bonuses */
        int outside = 0; /* penalty for king out of ranks [0,1] */
        int shield  = 0; /* pawn shield bonus */

        const auto occupied = state.occupied();

        for (auto color : { BLACK, WHITE })
        {
            const auto king = state.king(color);
            const auto ranks = color ? square_rank(king) : 7 - square_rank(king);
            const auto area = king_area(state, king);
            const auto color_mask = state.occupied_co(color);

            if (ranks > 1)
            {
                outside += SIGN[color] * (ranks - 1);
            }
            else
            {
                castle += SIGN[color]
                    * (ranks == 0)
                    * popcount(state.castling_rights & BB_BACKRANKS[color]);

                if (const auto pawns = state.pawns & color_mask)
                {
                    shield += SIGN[color]
                        * bool(BB_FILES[square_file(king)] & pawns)
                        * popcount(area & pawns);
                }
            }

            /*
             * https://www.chessprogramming.org/King_Safety  Attacking King Zone
             */
            static constexpr int ATTACK_WEIGHT[8] = { 0, 0, 50, 75, 88, 94, 97, 99 };

            for_each_square(area & ~color_mask, [&](Square square) {
                double attacks_value = 0;

                const auto attackers_mask =
                    state.attacker_pieces_mask(!color, square, occupied);

                for_each_square(attackers_mask, [&](Square attacking_square) {
                    const auto pt = state.piece_type_at(attacking_square);
                    ASSERT(pt > PieceType::PAWN && pt < PieceType::KING);

                    attacks_value += double(WEIGHT[pt] - 100) / KING_ATTACK_DIV;
                });

                const auto attackers = std::min(popcount(attackers_mask), 7);
                attacks -= SIGN[color] * attacks_value * ATTACK_WEIGHT[attackers] / 100;
            });
        }

        return attacks
            + eval_king_quadrant(state, pcs)
            + castle * interpolate(pcs, CASTLING_RIGHTS_BONUS, 0)
            + outside * interpolate(pcs, KING_OUT_PENALTY, 0)
            + shield * interpolate(pcs, PAWN_SHIELD, 0);
    }


    static INLINE int eval_material_imbalance(const State& state, int pcs)
    {
        if ((state.rooks | state.queens) == 0)
        {
            const int pawn_cnt[] = {
                popcount(state.pawns & state.occupied_co(BLACK)),
                popcount(state.pawns & state.occupied_co(WHITE))
            };

            for (auto side : { BLACK, WHITE })
            {
                if (pawn_cnt[side] < pawn_cnt[!side]
                    && ((state.bishops | state.knights) & state.occupied_co(side))
                   )
                {
                    return SIGN[side] * MATERIAL_IMBALANCE;
                }
            }
        }
        return 0;
    }


    /*
     * If the difference in material is within a pawn or less, favor
     * the side with two minor pieces over the side with extra rook.
     */
    static INLINE int eval_redundant_rook(const State& state, int pcs)
    {
        int score = 0;

        for (auto color : { BLACK, WHITE })
        {
            score += SIGN[color]
                * (popcount((state.bishops | state.knights) & state.occupied_co(!color)) >= 2)
                * (popcount(state.rooks & state.occupied_co(color)) >= 2);
        }

        return score * interpolate(pcs, 0, REDUNDANT_ROOK);
    }


    static INLINE int eval_open_files(const State& state, int piece_count)
    {
        static constexpr Bitboard opposite_backranks[] = {
            BB_RANK_2 | BB_RANK_1,
            BB_RANK_7 | BB_RANK_8,
        };

        int open_score = 0;
        int half_open_score = 0;

        for (auto color : { BLACK, WHITE })
        {
            const auto s = SIGN[color];
            const auto own_color_mask = state.occupied_co(color);

            for (const auto file_mask : BB_FILES)
            {
                if (auto mask = file_mask & (state.rooks | state.queens) & own_color_mask)
                {
                    if ((file_mask & state.pawns) == BB_EMPTY)
                    {
                        open_score += s;
                    }
                    else
                        if ((file_mask & state.pawns & own_color_mask) == BB_EMPTY)
                        {
                            half_open_score += s;
                        }
                }
            }
        }
        return half_open_score * interpolate(piece_count, MIDGAME_HALF_OPEN_FILE, 0)
             + open_score * interpolate(piece_count, MIDGAME_OPEN_FILE, 0);
    }


    static INLINE int eval_passed_formation(const State& state, int piece_count)
    {
        const auto diff =
            state.longest_pawn_sequence(state.occupied_co(WHITE) & (BB_RANK_6 | BB_RANK_7)) -
            state.longest_pawn_sequence(state.occupied_co(BLACK) & (BB_RANK_2 | BB_RANK_3));

        return diff * interpolate(piece_count, MIDGAME_PASSED_FORMATION, ENDGAME_PASSED_FORMATION);
    }


    static INLINE Bitboard pawn_defenders(const State& state, Color color, Square square)
    {
        return state.occupied_co(color) & state.pawns & BB_PAWN_ATTACKS[!color][square];
    }


    static INLINE int eval_pawn_chain(const State& state, Color color, Square pawn, int (&pawn_chain_evals)[64])
    {
        int& score = pawn_chain_evals[pawn];

        if (!is_valid(score))
        {
            score = 0;

            if (const auto defenders = pawn_defenders(state, color, pawn))
            {
                for_each_square(defenders, [&](Square square) {
                    score += 1 + eval_pawn_chain(state, color, square, pawn_chain_evals);
                });
            }
        }
        return score;
    }


    template<int i>
    static INLINE int eval_passed_pawns(const State& state, int piece_count, int (&pawn_chain_evals)[2][64])
    {
        struct PassedPawnRank
        {
            Bitboard mask[2]; /* midgame, endgame */
            int bonus[2]; /* ditto */
        };

        static
    #if TUNING_ENABLED || TUNING_PARTIAL
            const
    #else
            constexpr
    #endif
        PassedPawnRank ranks[2] = {
            {   /* 6th rank */
                { BB_RANK_3, BB_RANK_6 },
                { MIDGAME_UNBLOCKED_PASSED_6, ENDGAME_UNBLOCKED_PASSED_6 }
            },
            {   /* 7th rank */
                { BB_RANK_2, BB_RANK_7 },
                { MIDGAME_UNBLOCKED_PASSED_7, ENDGAME_UNBLOCKED_PASSED_7 }
            },
        };

        int score = 0;
        const auto occupied = state.occupied();

        for (auto color : { BLACK, WHITE })
        {
            const auto sign = SIGN[color];
            const auto own_mask = state.occupied_co(color);
            const auto others_mask = state.occupied_co(!color);

            const auto pawns = state.pawns & own_mask & ranks[i].mask[color];

            for_each_square(pawns, [&](Square square) {
                const auto pawn_mask = BB_SQUARES[square];
                const auto advance_mask = color ? shift_up(pawn_mask) : shift_down(pawn_mask);

                if ((advance_mask & occupied) == 0)
                {
                    score += sign * interpolate(piece_count, ranks[i].bonus[0], ranks[i].bonus[1]);
                }
                else if (BB_PAWN_ATTACKS[color][square] & others_mask)
                {
                    score += sign * interpolate(piece_count, ranks[i].bonus[0], ranks[i].bonus[1]);
                }

                score += sign
                    * eval_pawn_chain(state, color, square, pawn_chain_evals[color])
                    * interpolate(piece_count, MIDGAME_DEFENDED_PASSED, ENDGAME_DEFENDED_PASSED);
            });
        }
        return score;
    }


    /*
     * https://www.chessprogramming.org/images/7/70/LittleChessEvaluationCompendium.pdf
     * Grading of Pieces, page 4
     */
    static INLINE int eval_piece_grading(const State& state, int pcs)
    {
        int score = 0;
        const int p = popcount(state.pawns);

    #if USE_VECTOR && !(MOBILITY_TUNING)

        using ix4 = int __attribute__((vector_size(4 * sizeof(int))));
        auto constexpr N = WEIGHT[KNIGHT];
        auto constexpr B = WEIGHT[BISHOP];
        auto constexpr R = WEIGHT[ROOK];
        auto constexpr Q = WEIGHT[QUEEN];

        static constexpr ix4 perc_w[4] = {
            {   2 * N,  0 * B, -3 * R,  -2 * Q, }, /* closed */
            {   2 * N,  1 * B, -2 * R,  -1 * Q, }, /* semi-closed */
            {  -2 * N,  3 * B,  2 * R,   4 * Q, }, /* semi-open */
            {  -3 * N,  4 * B,  2 * R,   6 * Q, }, /* open */
        };
        using ux4 = uint64_t __attribute__((vector_size(4 * sizeof(uint64_t))));
        const auto& g = perc_w[int(p > 4) + int(p > 8) + int(p > 12)];
        const ux4 b = { state.knights, state.bishops, state.rooks, state.queens };

        for (const auto color : { BLACK, WHITE })
        {
            const auto c_mask = state.occupied_co(color);
            const ux4 c = b & c_mask;
            ix4 p = { popcount(c[0]), popcount(c[1]), popcount(c[2]), popcount(c[3]) };
            p *= g;

            score += SIGN[color] * (
                __builtin_reduce_add(p) / 100
                + popcount(state.pawns * c_mask) * interpolate(pcs, 0, 3)
            );
        }
    #else
        static constexpr int percents[4][4] = {
            /*  n,  b,   r,  q */
            {   2,  0, -3,  -2, }, /* closed */
            {   2,  1, -2,  -1, }, /* semi-closed */
            {  -2,  3,  2,   4, }, /* semi-open */
            {  -3,  4,  2,   6, }, /* open */
        };
        const auto& grading = percents[int(p > 4) + int(p > 8) + int(p > 12)];

        for (const auto color : { BLACK, WHITE })
        {
            const auto color_mask = state.occupied_co(color);

            score += SIGN[color] * (
                + popcount(state.knights & color_mask) * WEIGHT[KNIGHT] * grading[0]
                + popcount(state.bishops & color_mask) * WEIGHT[BISHOP] * grading[1]
                + popcount(state.rooks & color_mask) * WEIGHT[ROOK] * grading[2]
                + popcount(state.queens & color_mask) * WEIGHT[QUEEN] * grading[3]
            ) / 100;

            score += SIGN[color] * popcount(state.pawns * color_mask) * interpolate(pcs, 0, 3);
        }
    #endif /* USE_VECTOR && !MOBILITY_TUNING */

        return score;
    }


    static INLINE int eval_pawn_structure(const State& state, int pc)
    {
        int eval = eval_passed_formation(state, pc);

        int pawn_chain_evals[2][64];
        std::fill_n(&pawn_chain_evals[0][0], 2 * 64, SCORE_MIN);

        eval += eval_passed_pawns<0>(state, pc, pawn_chain_evals);
        eval += eval_passed_pawns<1>(state, pc, pawn_chain_evals);

        int doubled = 0;
        int isolated = 0;
        int diff = 0;

        for (auto color : { BLACK, WHITE })
        {
            const auto sign = SIGN[color];

            if (const auto own_pawns = state.pawns & state.occupied_co(color))
            {
                for (const auto& bb_file : BB_FILES)
                {
                    auto n = popcount(own_pawns & bb_file);
                    doubled += sign * (n > 1) * (n - 1);
                }
                diff += sign * popcount(own_pawns);
                isolated += sign * state.count_isolated_pawns(color);
            }
        }

        return eval
            + doubled * interpolate(pc, MIDGAME_DOUBLED_PAWNS, ENDGAME_DOUBLED_PAWNS)
            + isolated * interpolate(pc, MIDGAME_ISOLATED_PAWNS, ENDGAME_ISOLATED_PAWNS)
            + (diff != 0) * SIGN[diff > 0] * interpolate(pc, MIDGAME_PAWN_MAJORITY, ENDGAME_PAWN_MAJORITY);
    }


    static INLINE int eval_threats(const State& state, int piece_count)
    {
        int diff = 0;
        const auto occupied = state.occupied();

        for (auto color : { BLACK, WHITE })
        {
            const auto sign = SIGN[color];
            for_each_square(state.occupied_co(color) & ~(state.pawns | state.kings), [&](Square square) {
                diff -= sign * popcount(state.attackers_mask(!color, square, occupied));
            });
        }
        return diff * interpolate(piece_count, MIDGAME_THREATS, ENDGAME_THREATS);
    }


    static INLINE int eval_tactical(const State& state, score_t mat_eval, int piece_count)
    {
        score_t eval = eval_center(state, piece_count);

        if (abs(mat_eval) < WEIGHT[PAWN])
        {
            eval += eval_material_imbalance(state, piece_count);
            eval += eval_redundant_rook(state, piece_count);
        }

        eval += eval_open_files(state, piece_count);
        eval += eval_pawn_structure(state, piece_count);
        eval += eval_piece_grading(state, piece_count);
        eval += state.diff_connected_rooks()
             * interpolate(piece_count, MIDGAME_CONNECTED_ROOKS, ENDGAME_CONNECTED_ROOKS);

        if (state.bishops)
        {
            eval += BISHOP_PAIR * state.diff_bishop_pairs();
        }
        eval += eval_threats(state, piece_count);
        eval += eval_king_safety(state, piece_count);

        return eval;
    }


    static INLINE int eval_tactical(Context& ctxt, score_t eval)
    {
        const auto& state = ctxt.state();
        const auto piece_count = popcount(state.occupied());

        /*
         * 2nd order evaluation is currently slow (and possibly inaccurate).
         * To mitigate, in midgame it is done only at lower plies, and only
         * if 1st order eval delta is within a 2-3 pawns margin. The idea is
         * that deeper search paths may not benefit as much from qualitative
         * positional evaluation anyway; and tactical advantages will rarely
         * overcome significant material deficits.
         */
        if (state.is_endgame() || (ctxt._ply < EVAL_LOW_DEPTH && abs(eval) < EVAL_MARGIN))
        {
            const auto mat_eval = ctxt.evaluate_material();

            eval += SIGN[state.turn] * eval_tactical(state, mat_eval, piece_count);
        }
        else
        {
            eval += SIGN[state.turn] * eval_piece_grading(state, piece_count);
        }

        ASSERT(eval < SCORE_MAX);

        return eval;
    }
#endif /* !WITH_NNUE */

    /*
     * Static evaluation has three components:
     * 1. base = material + pst + mobility
     * 2. tactical (positional)
     * 3. capture estimates (in Context::evaluate)
     * NOTE: when using NNUE for evaluation (default), 1 and 2 do not apply.
     */
    score_t Context::_evaluate()
    {
        _tt->_eval_depth = std::max(_ply, _tt->_eval_depth);

        if (!is_valid(_eval))
        {
        #if WITH_NNUE
            eval_nnue();
        #else
            /*
             * 1. Material + piece-squares + mobility
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


    /*
     * Fractional extensions: https://www.chessprogramming.org/Extensions
     * "[...] extension can be added that does not yet extend the search,
     * but further down the tree may cause an extension when another
     * fractional extension causes the net extension to exceed one ply."
     */
    void Context::extend()
    {
        if (_extension || depth() >= MIN_EXT_DEPTH)
        {
            /*
             * things that could add interestingness along the search path
             * "what is ultimately to be reduced must first be expanded" Lao Tzu
             */
            _extension += state().pushed_pawns_score;
            _extension += _move.from_square() == _parent->_capture_square;
            _extension += is_recapture() * (is_pv_node() * (ONE_PLY - 1) + 1);

            /*
             * extend if move has historically high cutoff percentages and counts
             */
            _extension += ONE_PLY
                * (_move == _parent->_tt_entry._hash_move)
                * (abs(_parent->_tt_entry._value) < MATE_HIGH)
                * (_parent->history_count(_move) > HISTORY_COUNT_HIGH)
                * (_parent->history_score(_move) > HISTORY_HIGH);

            const auto double_extension_ok = (_double_ext <= DOUBLE_EXT_MAX);
            const auto extend = std::min(1 + double_extension_ok, _extension / ONE_PLY);

            ASSERT(extend >= 0);

            _max_depth += extend;
            _extension %= ONE_PLY;
            _double_ext += extend > 1;
        }

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
        return WINDOW_HALF * pow2(iteration) + WINDOW_COEFF * depth * log(0.001 + abs(score) / WINDOW_DIV);
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

            _alpha = std::max<score_t>(SCORE_MIN, score - std::max(WINDOW_HALF, delta));
            _beta = std::min<score_t>(SCORE_MAX, score + std::max(WINDOW_HALF, -delta));
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
        if (depth > 0 && count >= LMP[depth] * late_move_prune_factor() && can_prune())
            return LMRAction::Prune;

        /* no reductions at very low depth and in qsearch */
        if (depth < 3 || count < LATE_MOVE_REDUCTION_COUNT || !can_reduce())
            return LMRAction::None;

        /* Lookup reduction in the Late Move Reduction table. */
        auto reduction = LMR._table[std::min(depth, PLY_MAX-1)][std::min(count, 63)];

        if (_move._group != MoveOrder::TACTICAL_MOVES)
        {
            reduction += !_parent->has_improved();
            reduction -= 2 * _parent->is_counter_move(_move);

            if (get_tt()->_w_beta <= get_tt()->_w_alpha + 2 * WINDOW_HALF && iteration() >= 13)
                ++reduction;
            reduction -= _parent->history_count(_move) / HISTORY_COUNT_HIGH;
        }

        if (is_capture() || (_move.from_square() == _parent->_capture_square))
            --reduction;

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
            || state().promotion
            || is_check()
            /*
             * last move to search from current node, with score close to mate?
             * extend the search as to not miss a possible mate in the next move
             */
            || (_parent->_score < MATE_LOW && is_valid(_parent->_score) && is_last_move())
           )
            return false;

        /* treat it as leaf for now but retry and extend if it beats alpha */
        if (is_reduced())
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
        ensure_moves(ctxt);

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
                    ASSERT(move._group == MoveOrder::UNORDERED_MOVES);
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


    void MoveMaker::generate_unordered_moves(Context& ctxt)
    {
        /* pre-conditions */
        ASSERT(_count < 0);
        ASSERT(_current < 0);
        ASSERT(_phase == 0);
        ASSERT(_state_index == 0);

        auto& moves_list = ctxt.moves();

        auto& moves_cache = _moves_cache[ctxt.tid()];
        if (moves_cache.lookup(ctxt.state(), moves_list))
        {
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
            moves_cache.write(ctxt.state(), moves_list);
        }

        _count = int(moves_list.size());
        _current = 0;

        auto& states_vec = Context::states(ctxt.tid(), ctxt._ply);
        if (states_vec.size() < size_t(_count))
            states_vec.resize(_count);

    #if GROUP_QUIET_MOVES
        /*
         * In quiescent search, only quiet moves are interesting.
         * If in check, no need to determine "quieteness", since
         * all legal moves are about getting out of check.
         */
        _group_quiet_moves = (ctxt.depth() < 0 && !ctxt.is_check());
    #endif /* GROUP_QUIET_MOVES */
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
                else if (move == ctxt._tt_entry._best_move)
                {
                    make_move<false>(ctxt, move, MoveOrder::BEST_MOVES);
                }
                else if (move == ctxt._tt_entry._hash_move)
                {
                    make_move<false>(ctxt, move, MoveOrder::HASH_MOVES, ctxt._tt_entry._value);
                }
                else if (move.promotion())
                {
                    make_move<false>(ctxt, move, MoveOrder::PROMOTIONS, WEIGHT[move.promotion()]);
                }
                else if ((move.to_square() == ctxt._move.to_square() || ctxt.state().is_en_passant(move))
                    && make_move<false>(ctxt, move, MoveOrder::LAST_MOVED_CAPTURE))
                {
                    ASSERT(move._state->capture_value);
                    /*
                     * Looking at the capture of the last piece moved by the opponent before
                     * other captures may speed up the refutation of the current variation.
                     */

                    /* Sort in decreasing order of the capturing piece's value. */
                    move._score = -ctxt.state().piece_weight_at(move.from_square());
                }
            }
            /* Captures and killer moves. */
            else if constexpr (Phase == 2)
            {
                if (move._state ? move._state->capture_value : ctxt.state().is_capture(move))
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
                else if (move._score >= HISTORY_LOW
                    && make_move<true>(ctxt, move, futility)
                    && (move._state->has_fork(!move._state->turn) || is_direct_check(move)))
                {
                    move._group = MoveOrder::TACTICAL_MOVES;
                    move._score = hist_score;
                }
            }
            /* Phase == 4 */
            else if (make_move<true>(ctxt, move, futility))
            {
                incremental_update(move, ctxt);
                move._group = MoveOrder::LATE_MOVES;
                move._score =
                    ctxt.history_score(move) / (1 + HISTORY_LOW)
                    + eval_material_and_piece_squares(*move._state);
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
} /* namespace */


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
