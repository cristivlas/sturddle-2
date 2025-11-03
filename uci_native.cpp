/** Native C++ UCI */
/** http://wbec-ridderkerk.nl/html/UCIProtocol.html */
#include <filesystem>
#include <unordered_map>
#include "context.h"
#if WITH_NNUE
    #include "nnue.h"
#endif

namespace fs = std::filesystem;
using Params = std::unordered_map<std::string, std::string>;

#if NATIVE_UCI /* requires compiler with C++20 support */
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>
#if _WIN32
  #include <tlhelp32.h>
#else
  #include <unistd.h>
  #include <sys/resource.h>
#endif /* !_WIN32 */

#include "book.h"
#include "thread_pool.hpp" /* pondering, go infinite */

#if 0
  /* enable additional logging */
  #define LOG_DEBUG(x) while (_debug) { log_debug((x)); break; }
#else
  #define LOG_DEBUG(x)
#endif

#if _WIN32
static constexpr bool DEFAULT_HIGH_PRIORITY = true;
#else
static constexpr bool DEFAULT_HIGH_PRIORITY = false;
#endif /* !_WIN32 */

using ThreadPool = thread_pool<>;

static constexpr size_t initial_thread_count = 0;

static auto _compute_pool(std::make_unique<ThreadPool>(initial_thread_count));

static constexpr auto INFINITE = -1;
static std::string g_out; /* global output buffer */

namespace std
{
    INLINE std::string to_string(std::string_view v)
    {
        return std::string(v);
    }

#if defined(_MSVC_STL_VERSION) && __cpp_lib_format <= 202110L
    template<typename... Args>
    using format_string = _Fmt_string<Args...>;
#endif
}

namespace
{
    static constexpr std::string_view START_POS{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"};
    static bool _debug = false; /* enable verbose logging */

    template <typename T>
    static void log_error(T err)
    {
        try
        {
            search::Context::log_message(LogLevel::ERROR, std::to_string(err));
        }
        catch (...)
        {
        }
    }

    template <typename T> static void log_debug(T msg)
    {
        search::Context::log_message(LogLevel::DEBUG, std::to_string(msg));
    }

    template <typename T> static void log_info(T info)
    {
        search::Context::log_message(LogLevel::INFO, std::to_string(info));
    }

    template <typename T> static void log_warning(T warn)
    {
        search::Context::log_message(LogLevel::WARN, std::to_string(warn));
    }

    template <typename T> static INLINE T &lowercase(T &s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](auto c) { return std::tolower(c); });
        return s;
    }

    template <typename T> static INLINE std::string join(std::string_view sep, const T &v)
    {
        std::ostringstream s;
        for (const auto &elem : v)
            (s.tellp() ? s << sep : s) << elem;
        return s.str();
    }

    INLINE void output(std::ostream& out, const std::string& str)
    {
        out.write(str.data(), str.size());
    }

    template <bool Flush = true, typename T>
    INLINE void output(T&& str)
    {
        if (_debug)
            log_debug(std::format("<<< {}", str));

        if constexpr (std::is_const_v<std::remove_reference_t<T>>)
        {
            output(std::cout, str);
            std::cout.write("\n", 1);
        }
        else
        {
            str += "\n";
            output(std::cout, str);
        }

        if constexpr(Flush)
            std::cout.flush();
    }

    /** Raise ValueError exception, and exit with error (see dtor of GIL_State) */
    template <typename... Args>
    [[noreturn]] void raise_value_error(std::format_string<Args...> fmt, Args&&... args)
    {
        throw std::invalid_argument(std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename T> INLINE int to_int(T v)
    {
        try
        {
            return std::stoi(std::string(v));
        }
        catch(const std::exception& e)
        {
            if (v == "false")
                return 0;
            if (v == "true")
                return 1;
            raise_value_error("to_int({}): {}", v, e.what());
        }
    }


    static bool can_change_priority()
    {
    #if _WIN32
        /*
         * Do not allow Windows users to change the default behavior.
         */
        return false;
    #else
        /*
         * On POSIX assume true, to avoid libcap dependency (or manually parsing /proc/self/status).
         * If geteuid() != 0 and CAP_SYS_NICE not set, set_high_priority fails and resets _high_priority
         */
        return true;
    #endif /* !_WIN32 */
    }


    struct Option
    {
        virtual ~Option() = default;

        /* if show_default is true, print the default, otherwise show the = value */
        virtual void print(std::ostream &, bool show_default = true) const = 0;
        virtual void set(std::string_view value) = 0;
    };

    struct OptionBase : public Option
    {
        const std::string _name;
        explicit OptionBase(const std::string &name) : _name(name) {}
        void print(std::ostream &out, bool) const override { out << _name << " "; }
    };

    struct OptionAlgo : public OptionBase
    {
        search::Algorithm &_algo;

        explicit OptionAlgo(search::Algorithm& algo) : OptionBase("Algorithm"), _algo(algo) {}
        void print(std::ostream &out, bool show_default) const override
        {
            OptionBase::print(out, show_default);
            if (show_default)
                out << "type combo default mtdf var mtdf var negascout var negamax";
            else
                out << "type combo = " << name(_algo) << " var mtdf var negascout var negamax";
        }
        std::string_view name(search::Algorithm algo) const
        {
            switch (algo)
            {
            case search::Algorithm::MTDF: return "mtdf";
            case search::Algorithm::NEGAMAX: return "negamax";
            case search::Algorithm::NEGASCOUT: return "negascout";
            }
            return "";
        }
        void set(std::string_view value) override
        {
            if (value == "mtdf") _algo = search::Algorithm::MTDF;
            else if (value == "negascout") _algo = search::Algorithm::NEGASCOUT;
            else if (value == "negamax") _algo = search::Algorithm::NEGAMAX;
        }
    };

    struct OptionBool : public OptionBase
    {
        const bool _default_val;
        bool &_b;

        OptionBool(const std::string &name, bool &b) : OptionBase(name), _default_val(b), _b(b)
        {
        }

        void print(std::ostream &out, bool show_default) const override
        {
            OptionBase::print(out, show_default);
            if (show_default)
                out << "type check default " << std::boolalpha << _default_val;
            else
                out << "type check = " << std::boolalpha << _b;
        }

        void set(std::string_view value) override
        {
            if (value == "true")
                _b = true;
            else if (value == "false")
                _b = false;
        }
    };

    struct OptionParam : public OptionBase
    {
        const Param _p;

        OptionParam(const std::string &name, const Param &param) : OptionBase(name), _p(param) {}

        void print(std::ostream &out, bool show_default) const override
        {
            OptionBase::print(out, show_default);

            if (show_default)
            {
                if (_p.min_val == 0 && _p.max_val == 1)
                {
                    out << "type check default " << std::boolalpha << bool(_p.default_val);
                }
                else if (_p.normal)
                {
                    const auto scaled_val = 2.0 * (_p.default_val - _p.min_val) / (_p.max_val - _p.min_val) - 1;
                    out << "type string default " << scaled_val;
                }
                else
                {
                    out << "type spin default " << _p.default_val << " min " << _p.min_val << " max " << _p.max_val;
                }
            }
            else
            {
                if (_p.min_val == 0 && _p.max_val == 1)
                {
                    out << "type check = " << std::boolalpha << bool(_p.val);
                }
                else if (_p.normal)
                {
                    const auto scaled_val = 2.0 * (_p.val - _p.min_val) / (_p.max_val - _p.min_val) - 1;
                    out << "type string = " << scaled_val;
                }
                else
                {
                    out << "type spin = " << _p.val << " min " << _p.min_val << " max " << _p.max_val;
                }
            }
        }

        void set(std::string_view value) override
        {
            if (_p.normal)
            {
                const double v = std::stod(std::string(value));

                const auto val = std::round(((v + 1) / 2) * (_p.max_val - _p.min_val) + _p.min_val);
                _set_param(_name, int(val), true);
            }
            else
            {
                _set_param(_name, to_int(value), true);
            }
        }
    };

    struct OptionSyzygy : public OptionBase
    {
        OptionSyzygy() : OptionBase("SyzygyPath") {}

        void print(std::ostream& out, bool show_default) const override
        {
            OptionBase::print(out, show_default);
            out << "type string";

            if (show_default)
            {
                /* default setting is empty */
            }
            else
            {
                const auto &path = search::Context::syzygy_path();
                if (!path.empty()) out << " = " << path;
            }
        }

        void set(std::string_view value) override
        {
            search::Context::set_syzygy_path(std::string(value));
        }
    };
} /* namespace */


#if _WIN32
/*
 * Helpers for manage_console (see below).
 */
static DWORD get_parent_pid(DWORD processId)
{
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE)
        return 0;

    auto cleanup = on_scope_exit([hSnapshot]() {
        CloseHandle(hSnapshot);
    });

    PROCESSENTRY32 pe32 = {};
    pe32.dwSize = sizeof(PROCESSENTRY32);

    if (!Process32First(hSnapshot, &pe32))
        return 0;

    do {
        if (pe32.th32ProcessID == processId)
        {
            std::string name(pe32.szExeFile);

            // Running as a python script? bail
            if (lowercase(name).starts_with("python"))
                return 0;

            return pe32.th32ParentProcessID;
        }
    } while (Process32Next(hSnapshot, &pe32));

    return 0;
}

static bool ensure_console()
{
    if (!GetConsoleWindow())
    {
        if (!AllocConsole())
        {
            log_error(std::format("Could not allocate console, error: {}", GetLastError()));
        }
        else
        {
            // Rebind standard handles
            FILE* fp = nullptr;
            freopen_s(&fp, "CONIN$",  "r", stdin);
            freopen_s(&fp, "CONOUT$", "w", stdout);
            freopen_s(&fp, "CONOUT$", "w", stderr);

            return true;
        }
    }
    return false;
}


/*
 * An improved solution for: https://github.com/cristivlas/sturddle-2/issues/11
 *
 * Currently the engine runs from under the PyInstaller bootloader, and, under some
 * GUIs such as Shredder, an extra console pops up. The solution for recent Windows 11
 * builds is to use the "detached" setting in a manifest file at build time
 * (https://learn.microsoft.com/en-us/windows/console/console-allocation-policy).
 *
 * On older Windows versions: call FreeConsole if console detected in chess GUI mode.
 *
 * Return true if a console was allocated.
 */
static bool manage_console()
{
    /* Use STDIN handle to detect how the engine is being run. */
    const HANDLE h = GetStdHandle(STD_INPUT_HANDLE);

    DWORD mode = 0;

    if (GetConsoleMode(h, &mode))
    {
        if (auto wnd = GetConsoleWindow())
        {
            DWORD consolePID = 0;
            GetWindowThreadProcessId(wnd, &consolePID);

            const auto ourPID = GetProcessId(GetCurrentProcess());
            return (ourPID == consolePID) || get_parent_pid(ourPID) == consolePID;
        }
    }
    else if (GetFileType(h) == FILE_TYPE_PIPE)
    {
        /* STDIN is attached to a pipe, assume it is running under a chess GUI. */
        /* GUIs connect pipes to the engine's standard input and output to send */
        /* UCI command and to read back responses. */

        /* Do away with console window if detected. */
        if (GetConsoleWindow())
        {
            FreeConsole();
        }
    }
    else
    {
        /* The engine was likely started by the user double clicking in explorer.exe */
        /* or in some other file manager. The user likely wants to test the engine by */
        /* entering UCI commands manually, so make sure that there is a console. */
        return ensure_console();
    }
    return false;
}

#else

/* No action needed on POSIX. TODO: Test on Mac */
static bool manage_console()
{
    return false;
}
#endif /* _WIN32 */


class UCI
{
    using Arguments = std::vector<std::string_view>;
    using EngineOptions = std::map<std::string, std::unique_ptr<Option>>;

    static constexpr int max_depth = PLY_MAX;

public:
    UCI(const std::string &name, const std::string &version, Params& params)
        : _name(name)
        , _version(version)
    #if NATIVE_BOOK
        , _book((fs::absolute(fs::path(params["dir"])) / "book.bin").string())
        , _use_opening_book(true)
    #else
        , _use_opening_book(search::Context::_book_init(_book))
    #endif
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);

        search::Context::_history = std::make_unique<search::History>();

        log_debug(std::format("Context size: {}", sizeof(search::Context)));
        log_debug(std::format("ContextBuffer size: {}", sizeof(search::ContextBuffer)));
        log_debug(std::format("State size: {}", sizeof(chess::State)));
        log_debug(std::format("TT_Entry size: {}", sizeof(search::TT_Entry)));

        set_start_position();

        search::Context::_on_iter = on_iteration;
        search::Context::_on_move = on_move;

        refresh_options();
        if (_ponder)
            ensure_background_thread();

    #if 0 /* experimentation only */
        _options.emplace("algorithm", std::make_unique<OptionAlgo>(_algorithm));
    #endif
        _options.emplace("bestbookmove", std::make_unique<OptionBool>("BestBookMove", _best_book_move));
        _options.emplace("debug", std::make_unique<OptionBool>("Debug", _debug));
        _options.emplace("ownbook", std::make_unique<OptionBool>("OwnBook", _use_opening_book));
        _options.emplace("ponder", std::make_unique<OptionBool>("Ponder", _ponder));

    #if USE_ENDTABLES
        _options.emplace("syzygypath", std::make_unique<OptionSyzygy>());
    #endif /* USE_ENDTABLES */

        if (can_change_priority())
            _options.emplace("highpriority", std::make_unique<OptionBool>("HighPriority", _high_priority));
    }

    static bool output_expected() { return _output_expected.load(std::memory_order_relaxed); }
    void run();

private:
    void dispatch(std::string &, const Arguments &args);

    /** UCI commands */
    void debug();
    void go(const Arguments &args);
    void isready();
    void ponderhit();
    void position(const Arguments &args);
    void setoption(const Arguments &args);
    void stop();
    void uci();
    void newgame();

    void set_high_priority(bool);
    void show_settings();

    /** Context callbacks */
    static void on_iteration(PyObject *, search::Context *, const search::IterationInfo *);
    static void on_move(PyObject *, const std::string&, int);

private:
    void ensure_background_thread()
    {
        if (_compute_pool->get_thread_count() == 0)
        {
            _compute_pool = std::make_unique<ThreadPool>(1);
            log_info("Initialized UCI thread pool");
        }
    }

    /** position() helper */
    template <typename T> INLINE void apply_moves(const T &moves)
    {
        _last_move = chess::BaseMove();
        _ply_count = 0;

        search::Context::_history->emplace(_buf._state);

        for (const auto &m : moves)
            if (m.size() >= 4)
            {
                chess::Square from, to;

                if (chess::parse_square(m, from) && chess::parse_square(std::string_view(&m[2], 2), to))
                {
                    const auto promo = m.size() > 4 ? chess::piece_type(m[4]) : chess::PieceType::NONE;
                    const auto move = chess::BaseMove(from, to, promo);
                    const auto prev = _buf._state;
                    ASSERT(prev._hash == chess::zobrist_hash(prev));
                    _buf._state.apply_move(move);
                    chess::zobrist_update(prev, move, _buf._state);
                    ASSERT(_buf._state._hash == chess::zobrist_hash(_buf._state));
                    /* keep track of played moves, to detect repetitions */
                    search::Context::_history->emplace(_buf._state);
                    /* update the halfmove clock */
                    if (_buf._state.is_capture() || prev.piece_type_at(from) == chess::PieceType::PAWN)
                        search::Context::_history->_fifty = 0;
                    else
                        ++search::Context::_history->_fifty;
                    _last_move = move;
                    ++_ply_count;
                }
            }
        if (moves.empty())
            _buf._state.hash();
        ASSERT(_buf._state._hash);

        LOG_DEBUG(search::Context::epd(_buf._state));
    }

    INLINE search::Context &context() { return *_buf.as_context(true); }

    INLINE void output_best_move(bool request_ponder = false)
    {
        if (output_expected())
        {
            auto &ctxt = context();
            auto move = ctxt._best_move;
            if (!move)
                if (auto first = ctxt.first_valid_move())
                    move = *first;
            output_best_move(move, request_ponder);
        }
    }

    INLINE void output_best_move(const chess::BaseMove &move, bool request_ponder = false)
    {
        ASSERT(output_expected());
        _output_expected = false;

        if (!move)
        {
            output("bestmove 0000");
        }
        else
        {
            g_out.clear();
            if (request_ponder && _ponder)
            {
                const auto &pv = context().get_pv();
                if (pv.size() > 2 && pv[1] == move)
                {
                    std::format_to(std::back_inserter(g_out), "bestmove {} ponder {}", move.uci(), pv[2].uci());
                    output(g_out);
                    return;
                }
            }
            std::format_to(std::back_inserter(g_out), "bestmove {}", move.uci());
            output(g_out);
        }
    }

    INLINE chess::BaseMove search_book(bool validate = true)
    {
    #if NATIVE_BOOK
        if (!_opening_book.is_open())
        {
            log_debug(std::format("Opening: {}", _book));
            if (!_opening_book.open(_book))
            {
                _use_opening_book = false;
                log_error(std::format("Failed opening: {}", _book));
                return chess::BaseMove();
            }
        }

        const auto mode = _best_book_move ? PolyglotBook::BEST_WEIGHT : PolyglotBook::WEIGHTED_CHOICE;
        if (const auto raw_move = _opening_book.lookup_move(_buf._state.hash(), mode))
        {
            const auto move = chess::BaseMove(raw_move);

            if (!validate || _buf._state.is_valid(move))
            {
                LOG_DEBUG(std::format("Book move: {} in [{}]", move.uci(), search::Context::epd(_buf._state)));
                return move;
            }
            else
            {
                log_warning(std::format("Invalid book move: {} in [{}]", move.uci(), search::Context::epd(_buf._state)));
            }
        }
    #else
        /* Deprecated. Call into Python to look up Polyglot opening book. */
        if (const auto move = search::Context::_book_lookup(_buf._state, _best_book_move))
        {
            return move;
        }
    #endif /* !_NATIVE_BOOK */
        else
        {
            _book_depth = std::min(_book_depth, _ply_count);
        }

        return chess::BaseMove();
    }

    INLINE void set_start_position()
    {
        _buf._state = chess::State();
        _buf._state.castling_rights = chess::BB_DEFAULT_CASTLING_RIGHTS;
        chess::epd::parse_pos(START_POS, _buf._state);
        _buf._state.rehash();
    }

    void refresh_options()
    {
        for (auto p : _get_param_info())
        {
            auto name = p.first;
            /* option names are case insensitive, and can contain _single_ spaces */
            _options[lowercase(name)] = std::make_unique<OptionParam>(p.first, p.second);
        }
    }

    /** think on opponent's time */
    void ponder();

    /** iterative deepening search */
    template<typename F = void(*)()> score_t search(F f = []{});

    const std::string _name;
    const std::string _version; /* engine version */
    search::Algorithm _algorithm = search::Algorithm::MTDF;
    search::ContextBuffer _buf;
    search::TranspositionTable _tt;
    std::string _book = "book.bin";
    std::atomic_int _extended_time = 0; /* for pondering */
    int _book_depth = max_depth;
    int _depth = max_depth;
    int _ply_count = 0;
    score_t _score = 0;
    EngineOptions _options;
    static std::atomic_bool _output_expected;
    bool _ponder = false;
    bool _use_opening_book = false;
    bool _best_book_move = true;
    bool _current_priority = false; /* not high */
    bool _high_priority = DEFAULT_HIGH_PRIORITY;
    chess::BaseMove _last_move;
#if NATIVE_BOOK
    PolyglotBook _opening_book = {};
#endif
};

std::atomic_bool UCI::_output_expected(false);

/** Estimate number of moves (not plies!) until mate. */
static INLINE int mate_distance(score_t score, const search::PV &pv)
{
    return std::copysign((std::max<int>(CHECKMATE - std::abs(score), pv.size()) + 1) / 2, score);
}

/** Info sent to the GUI. */
struct Info : public search::IterationInfo
{
    static constexpr auto TIME_LOW = 1; /* millisec */

    const int eval_depth;
    const int hashfull;
    const int iteration;
    const bool brief;
    const search::PV* pv;
    static search::PV no_pv;

    Info(const search::Context& ctxt, const IterationInfo& info)
        : IterationInfo(info)
        , eval_depth(ctxt.get_tt()->_eval_depth)
        , hashfull(search::TranspositionTable::usage() * 10)
        , iteration(ctxt.iteration())
        , brief(milliseconds < TIME_LOW || !ctxt._best_move)
        , pv(brief ? &no_pv : &ctxt.get_pv())
    {}
};

search::PV Info::no_pv;

static void INLINE format_info(const Info& info)
{
    g_out.clear();
    if (info.brief)
    {
        std::format_to(std::back_inserter(g_out), "info score cp {} depth {}", info.score, info.iteration);
    }
    else
    {
        constexpr auto MATE_DIST_MAX = PV_PATH_MAX;

        auto score_unit = "cp";
        auto score = info.score;
        if (std::abs(info.score) > CHECKMATE - MATE_DIST_MAX)
        {
            score_unit = "mate";
            score = mate_distance(info.score, *info.pv);
        }
        std::format_to(
            std::back_inserter(g_out),
        #if USE_ENDTABLES
            "info score {} {} depth {} seldepth {} time {} nodes {} nps {} hashfull {} tbhits {} pv"
        #else
            "info score {} {} depth {} seldepth {} time {} nodes {} nps {} hashfull {} pv"
        #endif
            , score_unit
            , score
            , info.iteration
            , info.eval_depth
            , info.milliseconds
            , info.nodes
            , int(info.knps * 1000)
            , info.hashfull
        #if USE_ENDTABLES
            , info.tbhits
        #endif
            );

        /* output PV */
        for (size_t i = 1; i < info.pv->size(); ++i)
        {
            auto& m = (*info.pv)[i];
            if (!m)
                break;
            const auto uci = m.uci();
            g_out += " ";
            g_out += uci;
        }
    }
}


static void INLINE output_info(const Info& info)
{
    format_info(info);
    output<true>(g_out);
}


/* static */
void UCI::on_iteration(PyObject *, search::Context *ctxt, const search::IterationInfo *iter_info)
{
    if (ctxt && iter_info)
    {
        output_info(Info(*ctxt, *iter_info));
    }
}


/* static */
void UCI::on_move(PyObject *, const std::string& move, int move_num)
{
    static std::string move_info; /* thread-safe: callback is called at root, on thread 0 only */
    move_info.clear();
    std::format_to(std::back_inserter(move_info), "info currmove {} currmovenumber {}", move, move_num);
    output(move_info);
}


void UCI::run()
{
    Arguments args;
    std::string cmd;

    while (true)
    {
        std::getline(std::cin, cmd);
        if (std::cin.fail() || std::cin.eof())
        {
            stop();
            break;
        }
        const auto nl = cmd.find_last_not_of("\n\r");
        if (nl != std::string::npos)
            cmd.erase(nl + 1);
        if (cmd.empty())
            continue;
        if (_debug)
            log_debug(std::format(">>> {}", cmd));

        args.clear();
        /* tokenize command */
        std::ranges::for_each(
            std::views::lazy_split(cmd, std::string_view(" ")),
            [&](auto const &tok)
            {
                if (!tok.empty())
                    args.emplace_back(std::string_view(&*tok.begin(), std::ranges::distance(tok)));
            });

        if (args.empty())
            continue;

        if (args.front() == "quit")
        {
            _output_expected = false;
            stop();
            break;
        }
        dispatch(cmd, args);
    }
}

INLINE void UCI::dispatch(std::string &cmd, const Arguments &args)
{
    ASSERT(!args.empty());
    const auto& tok = args.front();
    switch (tok[0])
    {
    case 'd':
        if (tok == "debug")
        {
            debug();
            return;
        }
        break;
    case 'g':
        if (tok == "go")
        {
            go(args);
            return;
        }
        break;
    case 'i':
        if (tok == "isready")
        {
            isready();
            return;
        }
        break;
    case 'p':
        if (tok == "position")
        {
            position(args);
            return;
        }
        if (tok == "ponderhit")
        {
            ponderhit();
            return;
        }
        break;
    case 's':
        if (tok == "setoption")
        {
            setoption(args);
            return;
        }
        if (tok == "settings")
        {
            show_settings();
            return;
        }
        if (tok == "stop")
        {
            stop();
            return;
        }
        break;
    case 'u':
        if (tok == "uci")
        {
            uci();
            return;
        }
        if (tok == "ucinewgame")
        {
            newgame();
            return;
        }
        break;
    }
    log_error("unknown command: " + cmd);
}

template <typename T>
INLINE const auto &next(const T &v, size_t &i)
{
    static typename T::value_type empty;
    return ++i < v.size() ? v[i] : empty;
}

void UCI::debug()
{
#if _WIN32
    const bool use_unicode = (_isatty(_fileno(stdout)) && GetConsoleOutputCP() == CP_UTF8);
#else
    const bool use_unicode = true;
#endif
    search::Context::print_board(std::cout, _buf._state, use_unicode);
    output(std::format("fen: {}", search::Context::epd(_buf._state)));
    output(std::format("hash: {}", _buf._state._hash));
    size_t history_size = 0;
    history_size = search::Context::_history->_positions.size();
    output(std::format("history size: {}", history_size));
    output(std::format("halfmove clock: {}", search::Context::_history->_fifty));
    std::ostringstream checkers;
    chess::for_each_square(_buf._state.checkers_mask(_buf._state.turn),
        [&checkers](chess::Square sq) {
            checkers << sq << " ";
        });
    output(std::format("checkers: {}", checkers.str()));
}

void UCI::go(const Arguments &args)
{
    stop();

    bool explicit_movetime = false, do_analysis = false, do_ponder = false;
    int movestogo = 0, movetime = 0;
    double time_remaining[] = {0, 0};
    int time_increments[] = {0, 0};

    auto turn = _buf._state.turn;

    _depth = max_depth;

    for (size_t i = 1; i < args.size(); ++i)
    {
        const auto &a = args[i];
        if (a == "depth")
        {
            _depth = to_int(next(args, i));
            do_analysis = true;
        }
        else if (a == "movetime")
        {
            movetime = to_int(next(args, i));
            explicit_movetime = true;
        }
        else if (a == "movestogo")
        {
            movestogo = to_int(next(args, i));
        }
        else if (a == "wtime")
        {
            time_remaining[chess::WHITE] = to_int(next(args, i));
        }
        else if (a == "btime")
        {
            time_remaining[chess::BLACK] = to_int(next(args, i));
        }
        else if (a == "winc")
        {
            time_increments[chess::WHITE] = to_int(next(args, i));
        }
        else if (a == "binc")
        {
            time_increments[chess::BLACK] = to_int(next(args, i));
        }
        else if (a == "ponder")
        {
            do_ponder = true;
        }
        else if (a == "infinite")
        {
            movetime = -1;
            do_analysis = true;
        }
    }
    /* initialize search context */
    auto ctxt = new (_buf.as_context(false)) search::Context();
    _buf._valid = true;
    ctxt->_state = &_buf._state;

    if (!movetime)
        movetime = std::max<int>(0, time_remaining[turn] / std::max(movestogo, 40));
    LOG_DEBUG(std::format("movetime {}, movestogo {}", movetime, movestogo));

    _extended_time = 0;
    _output_expected = true;

    if (do_ponder)
    {
        _extended_time = std::max(1, movetime);
        ctxt->set_time_limit_ms(INFINITE);
        _compute_pool->push_task([this]{ ponder(); });
    }
    else if (do_analysis && !explicit_movetime)
    {
        ctxt->set_time_limit_ms(INFINITE);

        ensure_background_thread();
        _compute_pool->push_task([this]{
            search();
            output_best_move();
        });
    }
    else
    {
        if (_use_opening_book && _ply_count < _book_depth && !do_analysis)
        {
            LOG_DEBUG(std::format("lookup book_depth={}, ply_count={}", _book_depth, _ply_count));

            if (const auto move = search_book())
            {
                output_best_move(move);
                return;
            }
        }

        ASSERT(!do_analysis);
        ASSERT(!do_ponder);

        const auto set_time_limit = [&, ctxt] {
            if (explicit_movetime)
            {
                search::Context::set_time_limit_ms(movetime);
            }
            else
            {
                search::TimeControl ctrl;

                ctrl.millisec[chess::BLACK] = time_remaining[chess::BLACK];
                ctrl.millisec[chess::WHITE] = time_remaining[chess::WHITE];
                ctrl.increments[chess::BLACK] = time_increments[chess::BLACK];
                ctrl.increments[chess::WHITE] = time_increments[chess::WHITE];
                ctrl.moves = movestogo;
                ctrl.score = _score;

                search::Context::set_start_time();
                ctxt->set_time_ctrl(ctrl);
            }
        };
        /* search synchronously */
        _score = search(set_time_limit);
        /* Do not request to ponder below 100 ms per move. */
        output_best_move(movetime >= 100);
    }
}

/**
 * This command must always be answered with "readyok" and can be sent also
 * when the engine is calculating in which case the engine should also immediately
 * answer with "readyok" without stopping the search.
 */
INLINE void UCI::isready()
{
    output("readyok");
}

void UCI::newgame()
{
    stop();

    _tt.init(/* new_game = */ true);

    set_start_position();
    _book_depth = max_depth;

#if PST_TUNING_ENABLED
    chess::init_piece_square_tables();
    std::cout << "info string PST_init\n";
#endif
}

void UCI::set_high_priority(bool high_priority)
{
    if (_current_priority != high_priority)
    {
#if _WIN32
        const DWORD priority_class = high_priority ? HIGH_PRIORITY_CLASS : NORMAL_PRIORITY_CLASS;
        if (SetPriorityClass(GetCurrentProcess(), priority_class))
        {
            _current_priority = high_priority;
        }
        else
        {
            log_error(std::format("SetPriorityClass({}) failed: {}", priority_class, GetLastError()));
            _current_priority = _high_priority = false; /* prevent future calls */
        }
#else /* POSIX -- requires "sudo setcap cap_sys_nice+ep <engine_name>" */
        const int nice_value = high_priority ? -10 : 0;
        if (setpriority(PRIO_PROCESS, 0, nice_value) == 0)
        {
            _current_priority = high_priority;
        }
        else
        {
            const auto msg = std::format("setpriority({}) failed: {}", nice_value, errno);
            log_error(msg);
            std::cout << "info string " << msg << std::endl;
            _current_priority = _high_priority = false; /* prevent future calls */
        }
#endif /* !_WIN32 */
    }
}

void UCI::show_settings()
{
    std::cout << "*** Current Settings ***\n";

    refresh_options();

    for (const auto &opt : _options)
    {
        std::ostringstream opts;
        opt.second->print(opts << "option name ", false /* show current, not default*/);
        output<false>(opts.str());
    }
}

/**
 * Runs on a background thread with infinite time, and expects that:
 * either STOP is received; or
 * PONDERHIT is received, which extends the search by _extended_time,
 * then sets _extended_time to 0, to indicate to this function to send out
 * the best move when the search finishes.
 *
 * Pondering may finish before PONDERHIT is received, in which case
 * it resets _extended_time and does not output a move;
 *
 * the ponderhit handler will send out the best move instead, when PONDERHIT
 * is received (thus avoiding "premature bestmove in ponder" errors).
 */
void UCI::ponder()
{
    LOG_DEBUG(std::format("pondering, extended_time={}", _extended_time.load()));
    search();
    if (_extended_time)
        _extended_time = 0;
    else
        output_best_move();
}

void UCI::ponderhit()
{
    if (int ext = _extended_time)
    {
        _extended_time = 0;
        context().set_time_limit_ms(ext);
    }
    else
    {
        stop();
    }
}

void UCI::position(const Arguments &args)
{
    stop();

    bool in_moves = false;
    static Arguments fen, moves;

    fen.clear();
    moves.clear();
    search::Context::_history->clear();

    for (const auto &a : std::ranges::subrange(args.begin() + 1, args.end()))
    {
        if (a == "fen")
        {
            in_moves = false;
        }
        else if (a == "moves")
        {
            in_moves = true;
        }
        else if (a == "startpos")
        {
            set_start_position();
            in_moves = false;
        }
        else if (in_moves)
        {
            moves.emplace_back(a);
        }
        else
        {
            fen.emplace_back(a);
        }
    }
    if (fen.size() >= 4)
    {
        _buf._state = chess::State();
        ASSERT(_buf._state._hash == 0);

        if (   !chess::epd::parse_pos(fen[0], _buf._state)
            || !chess::epd::parse_side_to_move(fen[1], _buf._state)
            || !chess::epd::parse_castling(fen[2], _buf._state)
            || !chess::epd::parse_en_passant_target(fen[3], _buf._state)
           )
            raise_value_error("fen={} {} {} {}", fen[0], fen[1], fen[2], fen[3]);
    }
    else if (!fen.empty())
    {
        raise_value_error("invalid token count {}, expected 4", fen.size());
    }
    _buf._state.rehash();
    apply_moves(moves);
    LOG_DEBUG(search::Context::epd(_buf._state));
}

template<typename F>
INLINE score_t UCI::search(F set_time_limit)
{
    if (!search::Context::_history)
        search::Context::_history = std::make_unique<search::History>();

    _tt.init(/* new_game = */ false);

    auto& ctxt = context();
    ctxt.set_tt(&_tt);

    ctxt._algorithm = _algorithm;
    ctxt._max_depth = 1;
    ctxt._move = _last_move;

    set_high_priority(_high_priority);
    auto restore_priority = on_scope_exit([this] { set_high_priority(false); });

    set_time_limit();

#if NATIVE_BOOK && USE_BOOK_HINT
    if (_ply_count < 12)
        ctxt._prev = search_book(false /* do not validate */);
#endif /* USE_BOOK_HINT */

    return search::iterative(ctxt, _tt, _depth + 1);
}

void UCI::setoption(const Arguments &args)
{
    Arguments name, value, *acc = nullptr;

    for (const auto &a : std::ranges::subrange(args.begin() + 1, args.end()))
    {
        if (a == "name")
            acc = &name;
        else if (a == "value")
            acc = &value;
        else if (acc)
            acc->emplace_back(a);
    }

    auto opt_name = join(" ", name);
    auto iter = _options.find(lowercase(opt_name));
    if (iter != _options.end())
        iter->second->set(join(" ", value));
    else
        log_warning(__func__ + (": \"" + opt_name + "\": not found"));

    if (_ponder)
        ensure_background_thread();
}

void UCI::stop()
{
#if 0
    search::Context::set_time_limit_ms(0);
    _compute_pool->wait_for_tasks([] { search::Context::cancel(); });
#else
    search::Context::cancel();
    _compute_pool->wait_for_tasks();
#endif
    output_best_move();
}

void UCI::uci()
{
    output<false>(std::format("id name {} {}", _name, _version));
    output<false>("id author Cristian Vlasceanu");

    /* show available options */
    for (const auto &opt : _options)
    {
        std::ostringstream opts;
        opt.second->print(opts << "option name ");
        output<false>(opts.str());
    }
    output("uciok");
}

void uci_loop(Params params)
{
    const auto console_allocated = manage_console();

    const auto name = params["name"];
    const auto version = params["version"];
    const auto debug = params["debug"];

#if WITH_NNUE
    output<false>(std::format("{} {} {}", name, version, nnue::instrset));
#else
    output<false>(std::format("{} {}", name, version));
#endif /* WITH_NNUE */

    _debug = (debug == "true");
    std::string err;
    try
    {
        UCI uci(name, version, params);
        uci.run();
    }
    catch (const std::exception &e)
    {
        err = e.what();
    }
    catch (...)
    {
        err = "unknown exception";
    }
    if (!err.empty())
    {
        search::Context::log_message(LogLevel::ERROR, err);
        std::cerr << err << std::endl;

        if (console_allocated)
        {
            std::cerr << "Press Enter to close this console... ";
            std::cin.get();
        }
    }
}

#else

void uci_loop(Params params)
{
    throw std::runtime_error("Native UCI implementation is not enabled.");
}
#endif /* NATIVE_UCI */
