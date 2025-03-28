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
 * This file contains parameters that control the behavior of the
 * search and evaluation functions, and infrastructure for exposing
 * them to Python scripts for the purpose of tuning the engine.
 *
 * To expose ALL settings, compile with -DTUNING_ENABLED -DWEIGHT_TUNING_ENABLED
 *
 * To cherry-pick, replace DECLARE_VALUE with DECLARE_PARAM
 */

constexpr int ONE_PLY = 16; /* fractional extensions */
constexpr int PLY_MAX = 100;

constexpr int PLY_HISTORY_MAX = 20;

constexpr score_t SCORE_MIN = -30000;
constexpr score_t SCORE_MAX =  30000;

constexpr score_t CHECKMATE = SCORE_MAX - 1;

#if MTDF_CSTAR_BISECT
    constexpr score_t MATE_HIGH = SCORE_MAX / 2;
#else
    constexpr score_t MATE_HIGH = SCORE_MAX - PLY_MAX;
#endif /* MTDF_CSTAR_BISECT */

constexpr score_t MATE_LOW  = -MATE_HIGH;


#if REFCOUNT_PARAM
/* Instrumentation & debug: count parameter usage */
struct Val
{
    int _v;
    mutable count_t _refcount = 0;

    explicit Val(int v) : _v(v) {}
    Val& operator=(int v) { _v = v; return *this; }
    INLINE operator int() const { ++_refcount; return _v; }
};
#else
using Val = int;
#endif /* REFCOUNT_PARAM */

#if defined(CONFIG_IMPL)
#include <map>
#include <string>
#include <thread>

struct Config
{
    struct Param /* meta param info */
    {
        Val* const  _val = nullptr;
        const int   _min = 0;
        const int   _max = 0;
        std::string _group;
    };

    using Namespace = std::map<std::string, Param>;

    static Namespace _namespace;
    static std::string _group;

    /* Register parameter names with Config::_namespace */
    Config(const char* n, Val* v, int v_min, int v_max)
    {
        _namespace.emplace(n, Config::Param{ v, v_min, v_max, Config::_group });
    }

    struct Group
    {
        Group(const char* group) { _group.assign(group); }
        ~Group() { _group.clear(); }
    };
};

std::string Config::_group;
#define GROUP(x) Config::Group __##x(_TOSTR(x));

Config::Namespace Config::_namespace = {
#if MOBILITY_TUNING_ENABLED
    /*** DEPRECATE Piece mobility coefficients ***/
    { "MOBILITY_PAWN", Config::Param{ &chess::MOBILITY[chess::PieceType::PAWN], 0, 50, "Weights" } },
    { "MOBILITY_KNIGHT", Config::Param{ &chess::MOBILITY[chess::PieceType::KNIGHT], 0, 50, "Weights" } },
    { "MOBILITY_BISHOP", Config::Param{ &chess::MOBILITY[chess::PieceType::BISHOP], 0, 50, "Weights" } },
    { "MOBILITY_ROOK", Config::Param{ &chess::MOBILITY[chess::PieceType::ROOK], 0, 50, "Weights" } },
    { "MOBILITY_QUEEN", Config::Param{ &chess::MOBILITY[chess::PieceType::QUEEN], 0, 50, "Weights" } },
    { "MOBILITY_KING", Config::Param{ &chess::MOBILITY[chess::PieceType::KING], 0, 50, "Weights" } },
#endif /* MOBILITY_TUNING_ENABLED */

#if WEIGHT_TUNING_ENABLED
    /* Piece mobility coefficients */
    { "PAWN", Config::Param{ &chess::WEIGHT[chess::PieceType::PAWN], 0, 150, "Weights" } },
    { "KNIGHT", Config::Param{ &chess::WEIGHT[chess::PieceType::KNIGHT], 0, 400, "Weights" } },
    { "BISHOP", Config::Param{ &chess::WEIGHT[chess::PieceType::BISHOP], 0, 400, "Weights" } },
    { "ROOK", Config::Param{ &chess::WEIGHT[chess::PieceType::ROOK], 0, 700, "Weights" } },
    { "QUEEN", Config::Param{ &chess::WEIGHT[chess::PieceType::QUEEN], 0, 1300, "Weights" } },
#endif /* WEIGHT_TUNING_ENABLED */
};
#else

  #define GROUP(x) /* as nothing */

#endif /* !CONFIG_IMPL */

/*
 * Runtime params that are always visible (regardless of TUNING_ENABLED).
 */
#if !defined(CONFIG_IMPL)
  #define DECLARE_ALIAS(n, a, v, v_min, v_max) extern Val n;
#else
  #define DECLARE_ALIAS(n, a, v, v_min, v_max) Val n(v); Config p_##n(_TOSTR(a), &n, v_min, v_max);
#endif /* CONFIG_IMPL */

#define DECLARE_CONST(n, v, v_min, v_max) static constexpr int n = v; static_assert(v >= v_min && v <= v_max);
#define DECLARE_PARAM(n, v, v_min, v_max) DECLARE_ALIAS(n, n, v, v_min, v_max)

#if TUNING_ENABLED
  #define DECLARE_VALUE(n, v, v_min, v_max) DECLARE_PARAM(n, v, v_min, v_max)
#else
 /*
  * tuning disabled: params become compile-time constants
  */
  #define DECLARE_VALUE DECLARE_CONST

#endif /* TUNING_ENABLED */

static constexpr int HASH_MIN = 16; /* MB */

#if SMP
    #if defined(CONFIG_IMPL)
        static const auto THREAD_MAX = std::thread::hardware_concurrency();
        static const auto THREAD_VAL = std::min<int>(4, THREAD_MAX);
    #endif
#else
    static constexpr int SMP_CORES = 1;
#endif /* SMP */

/* Min-max range is useful when exposing params via UCI. */
/****************************************************************************
 *              NAME                                VALUE  MIN      MAX
 ****************************************************************************/

GROUP(Settings)
#if SMP
    DECLARE_ALIAS( SMP_CORES, Threads,         THREAD_VAL,  1,  THREAD_MAX)
#endif

DECLARE_VALUE(  ASPIRATION_WINDOW,                    1,    0,       1)
DECLARE_CONST(  DEBUG_CAPTURES,                       0,    0,       1)
#if DATAGEN
    DECLARE_CONST(  DATAGEN_SCORE_THRESHOLD,          0,    0,   30000)
    DECLARE_CONST(  DATAGEN_MIN_DEPTH,               10, -100,     100)
#endif
#if EVAL_FUZZ_ENABLED
    DECLARE_PARAM(  EVAL_FUZZ,                        0,    0,     100)
#endif
DECLARE_CONST(  FIFTY_MOVES_RULE,                     1,    0,       1)
DECLARE_VALUE(  FUTILITY_PRUNING,                     1,    0,       1)
DECLARE_VALUE(  MULTICUT,                             1,    0,       1)

GROUP(Search)
DECLARE_VALUE(  CAPTURES_SCALE,                      99,    0,     150)
DECLARE_VALUE(  CAPTURES_THRESHOLD,           MATE_HIGH,    0,   30000)
DECLARE_VALUE(  DOUBLE_EXT_MARGIN,                 1354,    0,    2000)
DECLARE_VALUE(  DOUBLE_EXT_MAX,                      12,    0,     100)
DECLARE_VALUE(  LATE_MOVE_REDUCTION_COUNT,            4,    0,     100)
DECLARE_VALUE(  LMP_ALPHA,                           17,    0,     100)
DECLARE_VALUE(  LMP_BETA,                            36,    1,     100)
DECLARE_VALUE(  LMP_BASE,                             2,    2,     100)
DECLARE_VALUE(  KILLER_MOVES_MIN_DEPTH,               1,    0,     100)
DECLARE_VALUE(  KILLER_MOVES_MARGIN,                261,    0,    1000)

DECLARE_VALUE(  MIN_EXT_DEPTH,                        7,    0,     100)
DECLARE_VALUE(  MULTICUT_MARGIN,                    124,    0,    1000)

#if WITH_NNUE
DECLARE_VALUE(  NNUE_EVAL_TERM,                     645,    0,    1000)
DECLARE_VALUE(  NNUE_MAX_EVAL,                      496,    0,    1000)
#endif /* WITH_NNUE */

DECLARE_VALUE(  NULL_MOVE_DEPTH_WEIGHT,               3,    0,     100)
DECLARE_VALUE(  NULL_MOVE_DEPTH_DIV,                  4,    1,     100)
DECLARE_VALUE(  NULL_MOVE_DIV,                      278,    1,    1000)
DECLARE_VALUE(  NULL_MOVE_REDUCTION,                  4,    0,     100)
DECLARE_VALUE(  NULL_MOVE_IMPROVEMENT_DIV,           72,    1,    1000)
DECLARE_VALUE(  NULL_MOVE_MARGIN,                   611,    0,    1000)
DECLARE_VALUE(  NULL_MOVE_MIN_VERIFICATION_DEPTH,    14,    0,     100)
DECLARE_VALUE(  RAZOR_DEPTH_COEFF,                  248,    0,     300)
DECLARE_VALUE(  RAZOR_INTERCEPT,                    224,    0,     300)
DECLARE_VALUE(  REBEL_EXTENSION,                      3,    1,       4)
DECLARE_VALUE(  REBEL_EXTENSION_MARGIN,              56,    0,     500)
DECLARE_VALUE(  REVERSE_FUTILITY_MARGIN,             33,    0,     150)
/* SEE */
DECLARE_VALUE(  SEE_PRUNING,                          1,    0,       1)
DECLARE_VALUE(  SEE_PRUNING_DEPTH,                    3,    1,      20)
/* -1 disables pin awareness */
DECLARE_VALUE(  SEE_PIN_AWARENESS_DEPTH,             -1,   -1,     100)

DECLARE_VALUE(  SINGULAR_ACCURACY_MARGIN,           288,    1,     500)
DECLARE_VALUE(  SINGULAR_DEPTH_MARGIN,                0,    0,     100)

DECLARE_CONST(  STATIC_EXCHANGES,                     0,    0,       1)
DECLARE_VALUE(  STANDPAT_MARGIN,                     85,    0,    1000)

DECLARE_VALUE(  TIME_CTRL_EVAL_THRESHOLD_LOW,       -48, -150,       0)
DECLARE_VALUE(  TIME_CTRL_EVAL_THRESHOLD_HIGH,       12,    0,     150)

/* Aspiration window */
DECLARE_VALUE(  WINDOW_COEFF,                         6,    0,     100)
DECLARE_VALUE(  WINDOW_DIV,                          67,    1,     200)
DECLARE_VALUE(  WINDOW_HALF,                         25,    5,     200)

GROUP(MoveOrdering)
DECLARE_VALUE(  COUNTER_MOVE_BONUS,                 223,    0,     500)
DECLARE_VALUE(  COUNTER_MOVE_MIN_DEPTH,               3,    0,      20)
DECLARE_VALUE(  HISTORY_COUNT_HIGH,               88415,    1,  100000)
DECLARE_VALUE(  HISTORY_SCORE_DIV,                   86,    1,     200)
DECLARE_VALUE(  HISTORY_SCORE_MUL,                  259,    0,    1000)
DECLARE_VALUE(  HISTORY_HIGH,                        92,    0,     100)
DECLARE_VALUE(  HISTORY_LOW,                         65,    0, HISTORY_HIGH)
DECLARE_VALUE(  HISTORY_MIN_DEPTH,                    3,    0,     100)
DECLARE_VALUE(  HISTORY_PRUNE,                       67,    0,     100)
/****************************************************************************/
#if !WITH_NNUE
/* HCE tunable parameters */
GROUP(Eval)
DECLARE_VALUE(  BISHOP_PAIR,                         53,    0,     100)
DECLARE_VALUE(  CASTLING_RIGHTS_BONUS,               32,    0,     100)
DECLARE_VALUE(  CENTER_ATTACKS,                      65,    0,     100)
DECLARE_VALUE(  CENTER_OCCUPANCY,                    60,    0,     100)
DECLARE_VALUE(  EVAL_MARGIN,                        300,    0,    5000)
DECLARE_VALUE(  EVAL_LOW_DEPTH,                       5,    0,     100)
DECLARE_VALUE(  KING_ATTACK_DIV,                     48,    1,     100)
DECLARE_VALUE(  KING_OUT_PENALTY,                  -120, -500,       0)
DECLARE_VALUE(  PAWN_SHIELD,                         21,    0,     100)
DECLARE_VALUE(  MATERIAL_IMBALANCE,                -235, -500,       0)
DECLARE_VALUE(  REDUNDANT_ROOK,                    -302, -500,       0)

DECLARE_VALUE(  ENDGAME_CONNECTED_ROOKS,             39,    0,     100)
DECLARE_VALUE(  ENDGAME_DEFENDED_PASSED,             29,    0,     100)
DECLARE_VALUE(  ENDGAME_KING_QUADRANT,               22,    0,     100)
DECLARE_VALUE(  ENDGAME_DOUBLED_PAWNS,              -42, -100,       0)
DECLARE_VALUE(  ENDGAME_ISOLATED_PAWNS,             -38, -100,       0)
DECLARE_VALUE(  ENDGAME_PASSED_FORMATION,            81,    0,     250)
DECLARE_VALUE(  ENDGAME_PAWN_MAJORITY,               65,    0,     250)
DECLARE_VALUE(  ENDGAME_THREATS,                     74,    0,     250)
DECLARE_VALUE(  ENDGAME_UNBLOCKED_PASSED_6,         148,    0,     250)
DECLARE_VALUE(  ENDGAME_UNBLOCKED_PASSED_7,         227,    0,     500)

DECLARE_VALUE(  MIDGAME_CONNECTED_ROOKS,             48,    0,     100)
DECLARE_VALUE(  MIDGAME_DEFENDED_PASSED,             68,    0,     100)
DECLARE_VALUE(  MIDGAME_KING_QUADRANT,               18,    0,     100)
DECLARE_VALUE(  MIDGAME_DOUBLED_PAWNS,              -28, -100,       0)
DECLARE_VALUE(  MIDGAME_ISOLATED_PAWNS,             -36, -100,       0)
DECLARE_VALUE(  MIDGAME_HALF_OPEN_FILE,              48,    0,     250)
DECLARE_VALUE(  MIDGAME_OPEN_FILE,                   65,    0,     250)
DECLARE_VALUE(  MIDGAME_PASSED_FORMATION,            79,    0,     250)
DECLARE_VALUE(  MIDGAME_PAWN_MAJORITY,              109,    0,     250)
DECLARE_VALUE(  MIDGAME_THREATS,                     83,    0,     250)
DECLARE_VALUE(  MIDGAME_UNBLOCKED_PASSED_6,         124,    0,     250)
DECLARE_VALUE(  MIDGAME_UNBLOCKED_PASSED_7,         163,    0,     250)
#endif /* !WITH_NNUE */
/****************************************************************************/
#undef DECLARE_ALIAS
#undef DECLARE_PARAM
#undef DECLARE_VALUE
#undef GROUP
