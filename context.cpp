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
#include "common.h"
/*
 * Move ordering, board state evaluation, and other stuff
 * pertaining to the Context of the node being searched.
 */
#include <cerrno>
#include <iomanip>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include "chess.h"

#define CONFIG_IMPL
  #include "context.h"
#undef CONFIG_IMPL

#if !defined(WITH_NNUE)
  #define WITH_NNUE true
#endif

#if WITH_NNUE
  #include "nnue.h"
  #if !(SHARED_WEIGHTS)
    #include "weights.h"
  #endif
#endif

#include "eval.h"

#if USE_ENDTABLES
  #include "tbprobe.h"
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

            for (size_t j = 0; j < BUCKET_SIZE; ++j)
            {
                const auto i = (hash + j) & (_data.size() - 1);
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

            for (size_t j = 0; j < BUCKET_SIZE; ++j)
            {
                const auto i = (hash + j) & (_data.size() - 1);
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
                _table[depth][moves] = 0.5 + log(depth) * log(moves) / M_E;
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
                elem.second._default_val,
                elem.second._min,
                elem.second._max,
                elem.second._group,
                elem.second._normal
            });
    }

    info.emplace("Hash", Param {
        int(TranspositionTable::get_hash_size()),
        DEFAULT_HASH_TABLE_SIZE,
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
    else if (name == "Threads" && value != SMP_CORES)
    {
        search::stop_threads();
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

constexpr int INPUTS_A = 3588;
constexpr int INPUTS_B = 256;
constexpr int HIDDEN_1A = 1280;
constexpr int HIDDEN_1A_POOLED = HIDDEN_1A / nnue::POOL_STRIDE;
constexpr int HIDDEN_1B = 64;
constexpr int HIDDEN_2 = 16;
constexpr int HIDDEN_3 = 16;

using LAttnType = nnue::Layer<HIDDEN_1B * nnue::ATTN_BUCKETS, 32>;
using L1AType = nnue::Layer<INPUTS_A, HIDDEN_1A, int16_t, nnue::QSCALE, true /* incremental */>;
using L1BType = nnue::Layer<INPUTS_B, HIDDEN_1B, int16_t, nnue::QSCALE, true /* incremental */>;
using L2Type = nnue::Layer<HIDDEN_1A_POOLED, HIDDEN_2>;
using L3Type = nnue::Layer<HIDDEN_2, HIDDEN_3>;
using EVALType = nnue::Layer<HIDDEN_3, 1>;

/*
 * The accumulator takes the inputs and processes them into two outputs,
 * using layers L1A and L1B. L1B processes the 1st 256 inputs, which
 * correspond to kings and pawns. The output of L1B is processed by the
 * spatial attention layer, which moodulates the outputs of the L1A layer.
 */
using Accumulator = nnue::Accumulator<INPUTS_A, HIDDEN_1A, HIDDEN_1B>;
using AccumulatorStack = std::array<Accumulator, PLY_MAX>;

using LMOVEType = nnue::Layer<INPUTS_A / Accumulator::NUM_BUCKETS, 4096, int16_t, nnue::QSCALE>;

/* Each thread uses its own stack */
static std::vector<AccumulatorStack> NNUE_data(SMP_CORES);

static struct Model
{
    void init();

    void validate_weights_file(const std::filesystem::path& weights_path)
    {
        constexpr auto param_count =
            L1AType::param_count()
            + L1BType::param_count()
            + LAttnType::param_count()
            + L2Type::param_count()
            + L3Type::param_count()
            + EVALType::param_count()
        #if USE_MOVE_PREDICTION
            + LMOVEType::param_count()
        #endif
            ;
        constexpr auto expected_size = param_count * sizeof(float);
        const auto file_size = std::filesystem::file_size(weights_path);
        if (file_size != expected_size)
            throw std::runtime_error(weights_path.string() + ": expected " + std::to_string(expected_size) + " bytes, got " + std::to_string(file_size));
    }

    void load_weights(const std::filesystem::path& weights_path)
    {
        validate_weights_file(weights_path);

        std::ifstream file(weights_path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Could not open weights file: " + weights_path.string());

        file.exceptions(std::ios::failbit | std::ios::badbit);

        try
        {
            /* Load layers in the same order that the trainer exports them. */
            L1B.load_weights(file);
            L1A.load_weights(file);
            LATTN.load_weights(file);
            L2.load_weights(file);
            L3.load_weights(file);
            EVAL.load_weights(file);

        #if USE_MOVE_PREDICTION
            LMOVES.load_weights(file);
        #endif
        }
        catch (const std::exception&)
        {
            throw std::runtime_error("Error reading weights from: " + weights_path.string());
        }
        Context::log_message(LogLevel::INFO, "Loaded " + weights_path.string());
    }

    std::string default_weights_path;

    LAttnType LATTN;
    L1AType L1A;
    L1BType L1B;
    L2Type L2;
    L3Type L3;
    EVALType EVAL;

#if USE_MOVE_PREDICTION
    LMOVEType LMOVES;
#endif

} model;


#if !SHARED_WEIGHTS
/* weights are compiled-in */
#define INIT_LAYER(layer, name) layer.set_weights(name ## _w, name ## _b)

void Model::init()
{
    INIT_LAYER(LATTN, spatial_attn);
    INIT_LAYER(L1A, hidden_1a);
    INIT_LAYER(L1B, hidden_1b);
    INIT_LAYER(L2, hidden_2);
    INIT_LAYER(L3, hidden_3);
    INIT_LAYER(EVAL, out);

#if USE_MOVE_PREDICTION
    INIT_LAYER(LMOVES, move);
#endif
}
#else

void Model::init()
{
    if (!default_weights_path.empty())
    {
        load_weights(default_weights_path);
    }
}
#endif /* SHARED_WEIGHTS */


static void _load_weights(const std::string& file_path)
{
    if (file_path.empty())
        model.init();
    else
        model.load_weights(file_path);
}


void Context::load_weights(const std::string& file_path)
{
    cython_wrapper::call_nogil(_load_weights, file_path); // catch exception and translate to Python
}


static INLINE void update(Accumulator& accumulator, const Context* ctxt)
{
    accumulator.update(model.L1A, model.L1B, ctxt->state());
}


/* incremental version */
static INLINE void update(Accumulator& accumulator, const Context* ctxt, Accumulator& prev_acc)
{
    accumulator.update(model.L1A, model.L1B, ctxt->_parent->state(), ctxt->state(), ctxt->_move, prev_acc);
}


void search::Context::update_accumulators()
{
    const auto t = tid();

    Context* update_chain[PLY_MAX];
    size_t chain_length = 0;

    // Collect contexts that need updates
    for (auto ctxt = this; ctxt; ctxt = ctxt->_parent)
    {
        auto& accumulator = NNUE_data[t][ctxt->_ply];
        if (!accumulator.needs_update(ctxt->state()))
            break;

        ASSERT(chain_length < PLY_MAX);
        update_chain[chain_length++] = ctxt;
    }

    // Update in reverse order
    for (auto i = chain_length; i > 0; --i)
    {
        auto* ctxt = update_chain[i - 1];
        auto& accumulator = NNUE_data[t][ctxt->_ply];

        if (ctxt->is_root())
        {
            ASSERT(ctxt->_parent == nullptr);
            update(accumulator, ctxt);
        }
        else
        {
            auto& prev_acc = NNUE_data[t][ctxt->_ply - ctxt->_nnue_prev_offs];
            ASSERT(!prev_acc.needs_update(ctxt->_parent->state()));

            update(accumulator, ctxt, prev_acc);
        }

        ctxt->_eval_raw = SCORE_MIN;
    }
}


score_t search::Context::eval_nnue_raw(bool stm_perspective)
{
    ASSERT(!is_valid(_eval_raw));

    update_accumulators();

    auto& acc = NNUE_data[tid()][_ply];
    ASSERT(!acc.needs_update(state()));

    _eval_raw = nnue::eval(acc, model.LATTN, model.L2, model.L3, model.EVAL);

    if (stm_perspective)
    {
        _eval_raw *= SIGN[state().turn];
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
            const auto eval_nn = eval_nnue_raw(true);

            eval = (eval_nn * NNUE_BLEND_PERCENT + eval * (100 - NNUE_BLEND_PERCENT)) / 100;
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

    #if MATERIAL_CORRECTION_HISTORY
        if (abs(eval) < MATE_HIGH)
        {
            const auto mat_key = state().mat_key();
            const auto bucket = nnue::get_bucket(state());
            const auto correction = _tt->material_correction(turn(), mat_key, bucket);
            eval += correction / MATERIAL_CORRECTION_GRAIN;
        }
    #endif /* MATERIAL_CORRECTION_HISTORY */

        _eval = eval + eval_fuzz();
    }
}


#if WITH_NNUE
int search::Context::get_bucket() const
{
    return nnue::get_bucket(state());
}
#endif


void search::Context::update_root_accumulators()
{
    const auto& root = NNUE_data[0][0];

    for (int i = 1; i != SMP_CORES; ++i)
    {
        NNUE_data[i][0] = root;
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

    return ctxt.eval_nnue_raw(false);
}
#endif /* WITH_NNUE */


namespace search
{
    /*---------------------------------------------------------------------
     * Context
     *---------------------------------------------------------------------*/
    atomic_bool Context::_cancel(false);
    int         Context::_tb_cardinality(0);
    bool        Context::_tb_initialized(false);
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
    size_t (*Context::_vmem_avail)() = nullptr;

    std::string Context::_syzygy_path;


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


    /* static */ void Context::init(const std::string& exe_dir)
    {
        setup_crash_handler();
        _init(); /* Init attack masks and other magic bitboards in chess.cpp */

    #if WITH_NNUE
    #if SHARED_WEIGHTS
        try
        {
            const auto weights_path = std::filesystem::absolute(std::filesystem::path(exe_dir) / "weights.bin");

            model.load_weights(weights_path);
            model.default_weights_path = weights_path.string();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            _exit(-1);
        }
    #else
        model.init();
    #endif /* SHARED_WEIGHTS */
    #endif /* WITH_NNUE */
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

        copy_search_path(*this, *ctxt);

        return ctxt;
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
     * Track the best score and move so far;
     * called from search right after: score = -negamax(*next_ctxt).
     * Returns: Cutoff if beta cutoff, Retry if re-search needed, None otherwise.
     */
    FailHigh Context::is_beta_cutoff(Context* next_ctxt, score_t score)
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
                    return FailHigh::None;
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
                        if constexpr(EXTRA_STATS)
                            ++_tt->_retry_reductions;

                        return FailHigh::Retry;
                    }
                    else if (next_ctxt->_retry_above_alpha == RETRY::PVS && score < _beta)
                    {
                        _retry_beta = -score;
                        return FailHigh::Retry;
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

                copy_search_path(*next_ctxt, *this);
            }
        }

        ASSERT(_alpha >= _score); /* invariant */

        return _alpha >= _beta ? FailHigh::Cutoff : FailHigh::None;
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

        ASSERT(ply < Context::MAX_MOVE);

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
    INLINE int do_captures(int tid, const State& state, score_t standpat_threshold)
    {
        ASSERT(!state.is_check(!state.turn)); /* expect legal position */

        static constexpr auto ply = FIRST_EXCHANGE_PLY;

        auto mask = state.occupied_co(!state.turn);
        if (state.en_passant_square != Square::UNDEFINED)
            mask |= BB_SQUARES[state.en_passant_square];

        mask &= ~state.kings;

        /*
         * Generate all pseudo-legal captures.
         */
        auto& moves = Context::moves(tid, ply);
        if (state.generate_pseudo_legal_moves(moves, mask).empty())
        {
            return 0;
        }

        bool standpat = true;

        /*
         * 1) Go over the captures and assign the victim value to each move score.
         */
        for (auto& move : moves)
        {
            ASSERT(state.piece_type_at(move.from_square()));

            if (state.en_passant_square == move.to_square())
            {
                if (BB_SQUARES[move.from_square()] & state.pawns)
                    move._score = WEIGHT[PAWN]; /* TODO: piece-square, piece grading */
            }
            else
            {
                move._score = state.piece_value_at(move.to_square(), !state.turn); /* victim value */
            }

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
            if (move._score == 0)
                continue;

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
            result = do_captures(ctxt.tid(), *state, standpat_threshold);
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
        rewind();
        return get_next_move(0 /* = no futility pruning */);
    }


    static const char* piece_symbol(chess::PieceType type, chess::Color color, bool unicode)
    {
        if (unicode)
            return chess::UNICODE_PIECE_SYMBOLS[color][type];
        else
            return chess::PIECE_SYMBOL[color][type];
    }


    /* static */
    void Context::print_board(std::ostream& out, const State& state, bool unicode)
    {
        for (int rank = 7; rank >= 0; --rank)
        {
            out << rank + 1 << ' ';

            for (int file = 0; file < 8; ++file)
            {
                const auto square = Square(rank * 8 + file);
                if (const auto piece_type = state.piece_type_at(square))
                {
                    const auto piece_color = state.piece_color_at(square);
                    out << piece_symbol(piece_type, piece_color, unicode) << ' ';
                }
                else
                {
                    out << ". ";
                }
            }
            out << "\n";
        }
        out << "  a b c d e f g h" << std::endl;
    }


    /*
     * Reinitialize top context at the start of a new iteration.
     */
    void Context::reset(bool force_reorder_moves, bool clear_best_move)
    {
        ASSERT(is_root());
        ASSERT(!_is_null_move);
        ASSERT(!_is_retry);
        ASSERT(!_is_singleton);
        ASSERT(_tt->_w_alpha <= _alpha);
        ASSERT(_retry_above_alpha == RETRY::None);
        ASSERT(_pruned_count == 0); // not pruning at root
        ASSERT(_prune_reason == PruneReason::PRUNE_NONE);
        ASSERT(_futility_pruning);
        ASSERT(_capture_square == Square::UNDEFINED); // no null-move at root

        ASSERT(_double_ext == 0); // no extensions at root
        ASSERT(_extension == 0);
        ASSERT(_excluded.is_none());

        // Expect default values for these flags as
        // they should never be modified at root
        ASSERT(_multicut_allowed == MULTICUT);
        ASSERT(_null_move_allowed[WHITE] == true);
        ASSERT(_null_move_allowed[BLACK] == true);

        if (clear_best_move)
            _best_move = BaseMove();
        _cancel = false;
        _can_forward_prune = -1;

        _counter_move = Move();
        _cutoff_move = Move();
        _has_singleton = false;

        _max_depth = iteration() + (turn() == chess::BLACK);

        _mate_detected = 0;

        _repeated = -1;

        _retry_beta = SCORE_MAX;

        rewind(force_reorder_moves);
    }


    static INLINE int window_delta(int iteration, int depth, double score)
    {
        return WINDOW_HALF * std::log(1 + iteration) + WINDOW_COEFF * depth * std::log(1 + abs(score) / WINDOW_DIV);
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
            _beta = std::min<score_t>(SCORE_MAX, score + WINDOW_HALF);
        }
        else if (score >= _tt->_w_beta)
        {
            _alpha = std::max<score_t>(SCORE_MIN, score - WINDOW_HALF);
            _beta = std::min<score_t>(SCORE_MAX, score + window_delta(iteration(), _tt->_eval_depth, score));
        }
        else
        {
            const score_t delta = score - prev_score;
            prev_score = score;

            _alpha = std::max<score_t>(SCORE_MIN, score - std::max(WINDOW_HALF, delta));
            _beta = std::min<score_t>(SCORE_MAX, score + std::max(WINDOW_HALF, -delta));
        }

        ASSERT(_alpha < _beta);

        /* save iteration bounds */
        _tt->_w_alpha = _alpha;
        _tt->_w_beta = _beta;

        _score = SCORE_MIN;
    }


    static INLINE float fast_log2(float x)
    {
        union { float f; uint32_t i; } u = { x };
        // Extract exponent directly from IEEE 754 representation
        const int exp = ((u.i >> 23) & 0xFF) - 127;
        u.i = (u.i & 0x007FFFFF) | 0x3F800000;
        const float m = u.f;
        const float t = m - 1.0f;
        return exp + t * (1.442695041f + t * (-0.721347520f + t * 0.240449173f));
    }


    /*
     * Late move reduction and pruning.
     * https://www.chessprogramming.org/Late_Move_Reductions
     */
    LMRAction Context::late_move_reduce(int count, int64_t time_left)
    {
        ASSERT(count > 0);
        ASSERT(!is_null_move());
        ASSERT(_parent);

        const int depth = this->depth();

        /* late move pruning */
        if (depth > 0 && count >= LMP[depth] && can_prune())
            return LMRAction::Prune;

        /* no reductions at very low depth and in qsearch */
        if (depth < 3 || count < LATE_MOVE_REDUCTION_THRESHOLD || !can_reduce())
            return LMRAction::None;

        count += _tt->_pass / 4;

        /* Lookup reduction in the Late Move Reduction table. */
        auto reduction = LMR._table[std::min(depth, PLY_MAX-1)][std::min(count, 63)];

        /* Adjust for time -- main thread only */
        if (time_left)
        {
            auto node_count = time_left * _tt->_nps * 0.001f;
            auto recip_log_branch = std::min(16u, _tt->_pass) * 0.1 / 16 + 0.5 * piece_count() / 32;
            auto affordable_depth = fast_log2(1 + node_count) * recip_log_branch;
            reduction = std::max(reduction, std::max<int>(0, depth - affordable_depth));
        }

        if (_move._group != MoveOrder::TACTICAL_MOVES)
        {
            reduction += !_parent->has_improved<THEM>();
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
            ASSERT(!is_repeated());

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


    int64_t Context::check_time_and_update_nps(int64_t* time_left)
    {
        const auto millisec = elapsed_milliseconds();

        /*
         * Update nodes-per-second for this thread.
         */
        if (millisec)
            _tt->set_nps((1000 * _tt->nodes()) / millisec);
        else
            _tt->set_nps(_tt->nodes());

        const auto t = time_limit();
        if (t >= 0 && millisec >= t)
        {
            cancel();
            return -1;
        }

        if (time_left)
        {
            *time_left = std::max<int64_t>(0, t - millisec);
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


    /* static */ void Context::tb_init()
    {
#if USE_ENDTABLES
        if (_tb_initialized)
        {
            ::tb_free();
            _tb_initialized = false;
            _tb_cardinality = 0;
        }
        if (!_syzygy_path.empty())
        {
            try
            {
                _syzygy_path = std::filesystem::canonical(_syzygy_path).string();
            }
            catch (const std::exception& e)
            {
                _syzygy_path = "";
                log_message(LogLevel::ERROR, e.what());
                return;
            }

            _tb_initialized = ::tb_init(_syzygy_path.c_str());
            _tb_cardinality = TB_LARGEST;
        }
        std::cout << "info string " << _syzygy_path << " cardinality " << _tb_cardinality << "\n";
#endif /* USE_ENDTABLES */
    }


    /*---------------------------------------------------------------------
     * MoveMaker
     *---------------------------------------------------------------------*/

    int MoveMaker::rewind(Context& ctxt, bool force_reorder)
    {
        if (_count < 0)
            return -1;

        /*
         * Pruning decisions depend on depth; rewind without reorder
         * should only happen at the same depth as when moves were ordered,
         * unless no pruning occurred.
         */
        if (!force_reorder && (_have_pruned_moves || _have_quiet_moves))
        {
            ASSERT(_reorder_depth == ctxt._max_depth);
        }

        if (force_reorder)
        {
            _phase = 0;
            _reorder_depth = ctxt._max_depth;

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

        _current = 0;
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
        _reorder_depth = ctxt._max_depth;

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

    #if USE_MOVE_PREDICTION
        int active[65];  // 32 pieces + 1 stm + 16 + 16 occupancy
        int active_count = 0;
    #endif /* USE_MOVE_PREDICTION */

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
                    make_move<false>(ctxt, move, MoveOrder::PREV_ITER);
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
                    {
                        ASSERT(move._score == decltype(move._score)(hist_score));
                    }
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
                    move._group = MoveOrder::LATE_MOVES;
                #if USE_MOVE_PREDICTION
                    if (ctxt.iteration() <= MOVE_PREDICTION_MAX_ITER)
                    {
                        if (active_count == 0)
                            nnue::for_each_active_input(ctxt.state(), [&](int idx) {
                                ASSERT(active_count < 65);
                                active[active_count++] = idx;
                            });

                        nnue::score_move(model.LMOVES, active, active_count, move);
                    }
                    else
                #endif /* USE_MOVE_PREDICTION */
                    {
                        incremental_update(move, ctxt);
                        const auto eval = eval_material_for_side_that_moved(*move._state, ctxt._state, move);
                        move._score = ctxt.history_score(move) / (1 + HISTORY_LOW) + eval;
                    }
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

    #if 0 && USE_MOVE_PREDICTION /* debug */
        if (_phase == 4 && ctxt.iteration() <= MOVE_PREDICTION_MAX_ITER)
        {
            for (const auto& m : moves_list)
            {
                if (m._group == MoveOrder::LATE_MOVES)
                    std::cout << m.uci() << ": " << m._score << " (" << float(m._score) / nnue::QSCALE << ")\n";
            }
            std::cout << "\n";
        }
    #endif /* USE_MOVE_PREDICTION */
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
