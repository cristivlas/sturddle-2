/*
 * Launcher for the native build: picks the best flavor for this CPU
 * among the sturddle-<version>* executables in its own directory,
 * identified by the STURDDLE_FLAVOR= tag that each one carries.
 *
 * If a bundle payload is appended to the launcher (see build.py),
 * the engine files are extracted to a temp directory first, and removed
 * when the engine exits.
 */
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common.h"
#include "utility.h"
#include "version.h"
#include "zstd/lib/zstd.h"

namespace fs = std::filesystem;

static_assert(std::endian::native == std::endian::little, "bundle format is little-endian");

static constexpr std::string_view TAG_PREFIX = "STURDDLE_FLAVOR=";
static constexpr std::string_view BASE_NAME = "sturddle-" STURDDLE_VERSION;
static constexpr std::string_view BUNDLE_MAGIC = "STURDBND";
static constexpr std::string_view TEMP_PREFIX = "sturddle-bundle-";

static void read_exact(std::istream& in, void* buf, size_t size)
{
    if (!in.read(static_cast<char*>(buf), std::streamsize(size)))
        throw std::runtime_error("bundle: unexpected end of file");
}

#if _WIN32
static fs::path self_path()
{
    std::wstring buf(32768, L'\0');
    buf.resize(OS_CALL(GetModuleFileNameW, nullptr, buf.data(), DWORD(buf.size())));
    return buf;
}

static std::string program_name(const fs::path& path)
{
    return path.extension() == ".exe" ? path.stem().string() : std::string();
}

/* the Authenticode signature, if any, comes after the bundle */
static uint64_t payload_end(std::istream& in, uint64_t file_size)
{
    IMAGE_DOS_HEADER dos;
    read_exact(in.seekg(0), &dos, sizeof(dos));
    if (dos.e_magic != IMAGE_DOS_SIGNATURE)
        throw std::runtime_error("bundle: bad DOS header");

    IMAGE_NT_HEADERS nt;
    read_exact(in.seekg(dos.e_lfanew), &nt, sizeof(nt));
    if (nt.Signature != IMAGE_NT_SIGNATURE)
        throw std::runtime_error("bundle: bad PE header");

    const auto& security = nt.OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_SECURITY];
    return security.Size ? security.VirtualAddress : file_size;
}

/* %TEMP% is per user */
static bool create_private_dir(const fs::path& path)
{
    return fs::create_directory(path);
}

static std::wstring args_after_program()
{
    const std::wstring_view cmd = ::GetCommandLineW();
    size_t i = 0;
    if (!cmd.empty() && cmd[0] == L'"')
    {
        i = cmd.find(L'"', 1);
        i = (i == std::wstring_view::npos) ? cmd.size() : i + 1;
    }
    else
    {
        while (i < cmd.size() && cmd[i] != L' ' && cmd[i] != L'\t')
            ++i;
    }
    return std::wstring(cmd.substr(i));
}

/*
 * No exec on Windows: run the engine as a child that shares our std handles,
 * wait for it, and pass its exit code through. The job object kills the
 * engine if the launcher is terminated.
 */
static int run(const fs::path& target, char**)
{
    auto cmd = L"\"" + target.wstring() + L"\"" + args_after_program();

    const auto job = OS_CALL(CreateJobObjectW, nullptr, nullptr);

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    OS_CALL(SetInformationJobObject, job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));

    /* start the engine inside the job, so it is never left running on its own */
    SIZE_T attrs_size = 0;
    ::InitializeProcThreadAttributeList(nullptr, 1, 0, &attrs_size);
    std::vector<char> attrs_buf(attrs_size);
    const auto attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attrs_buf.data());
    OS_CALL(InitializeProcThreadAttributeList, attrs, 1, 0, &attrs_size);
    on_scope_exit delete_attrs([attrs] { ::DeleteProcThreadAttributeList(attrs); });

    HANDLE jobs[] = { job };
    OS_CALL(UpdateProcThreadAttribute,
        attrs,
        0,
        PROC_THREAD_ATTRIBUTE_JOB_LIST,
        jobs,
        sizeof(jobs),
        nullptr,
        nullptr);

    STARTUPINFOEXW si = {};
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = ::GetStdHandle(STD_INPUT_HANDLE);
    si.StartupInfo.hStdOutput = ::GetStdHandle(STD_OUTPUT_HANDLE);
    si.StartupInfo.hStdError = ::GetStdHandle(STD_ERROR_HANDLE);
    si.lpAttributeList = attrs;

    PROCESS_INFORMATION pi = {};
    OS_CALL(CreateProcessW,
        target.c_str(),
        cmd.data(),
        nullptr,
        nullptr,
        TRUE,
        EXTENDED_STARTUPINFO_PRESENT,
        nullptr,
        nullptr,
        &si.StartupInfo,
        &pi);
    ::CloseHandle(pi.hThread);

    /* ignore Ctrl-C, leave it to the engine; after CreateProcess so it is not inherited */
    OS_CALL(SetConsoleCtrlHandler, nullptr, TRUE);

    OS_CALL_CHECK(DwordResult, WaitForSingleObject, pi.hProcess, WAIT_INFINITE);

    DWORD code = EXIT_FAILURE;
    OS_CALL(GetExitCodeProcess, pi.hProcess, &code);
    return int(code);
}
#else
#include <csignal>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static fs::path self_path()
{
    return fs::read_symlink("/proc/self/exe");
}

static std::string program_name(const fs::path& path)
{
    return path.filename().string();
}

static uint64_t payload_end(std::istream&, uint64_t file_size)
{
    return file_size;
}

/* not writable by others even with a lax umask, /tmp is shared */
static bool create_private_dir(const fs::path& path)
{
    if (::mkdir(path.c_str(), 0700) == 0)
        return true;
    if (errno != EEXIST)
        throw_last_error("mkdir");
    return false;
}

static constexpr int FORWARDED_SIGNALS[] = { SIGTERM, SIGHUP };
static pid_t child_pid = 0;

static void forward_signal(int sig)
{
    const int saved_errno = errno;
    ::kill(child_pid, sig);
    errno = saved_errno;
}

/*
 * Run the engine as a child rather than exec it, so that we are still
 * around to remove the extracted files when it exits.
 */
static int run(const fs::path& target, char** argv)
{
    const auto path = target.string();
    std::vector<char*> args = { const_cast<char*>(path.c_str()) };
    for (auto arg = argv + 1; *arg; ++arg)
        args.push_back(*arg);
    args.push_back(nullptr);

    /* Ctrl-C goes to the whole process group, leave it to the engine */
    std::signal(SIGINT, SIG_IGN);

    /* hold the forwarded signals until there is a child to forward them to */
    sigset_t forwarded;
    sigemptyset(&forwarded);
    for (const int sig : FORWARDED_SIGNALS)
        sigaddset(&forwarded, sig);
    OS_CALL_CHECK(PosixResult, sigprocmask, SIG_BLOCK, &forwarded, nullptr);

    const pid_t parent = ::getpid();
    const pid_t pid = OS_CALL_CHECK(PosixResult, fork);
    if (pid == 0)
    {
        /* exec keeps ignored signals ignored, and blocked ones blocked */
        std::signal(SIGINT, SIG_DFL);
        ::sigprocmask(SIG_UNBLOCK, &forwarded, nullptr);

        /* die with the launcher, like the job object does on Windows */
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != parent)
            ::_exit(EXIT_FAILURE);

        ::execv(path.c_str(), args.data());
        std::println(stderr, "execv {} failed: {}", path, std::strerror(errno));
        ::_exit(EXIT_FAILURE);
    }

    /* let the engine exit on its own terms, then clean up */
    child_pid = pid;
    struct sigaction action = {};
    action.sa_handler = forward_signal;
    action.sa_flags = SA_RESTART;
    for (const int sig : FORWARDED_SIGNALS)
        OS_CALL_CHECK(PosixResult, sigaction, sig, &action, nullptr);
    OS_CALL_CHECK(PosixResult, sigprocmask, SIG_UNBLOCK, &forwarded, nullptr);

    int status = 0;
    OS_CALL_CHECK(PosixResult, waitpid, pid, &status, 0);

    /* nothing to forward to anymore, and the pid may get reused */
    for (const int sig : FORWARDED_SIGNALS)
        std::signal(sig, SIG_DFL);

    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
#endif /* _WIN32 */


#if defined(__x86_64__) || defined(_M_X64)
struct CpuId
{
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

static CpuId cpuid(uint32_t leaf, uint32_t subleaf = 0)
{
    int regs[4];
    __cpuidex(regs, int(leaf), int(subleaf));
    return { uint32_t(regs[0]), uint32_t(regs[1]), uint32_t(regs[2]), uint32_t(regs[3]) };
}

static uint64_t xgetbv0()
{
    return _xgetbv(0);
}
#else
#include <cpuid.h>

static CpuId cpuid(uint32_t leaf, uint32_t subleaf = 0)
{
    CpuId r;
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
    return r;
}

static uint64_t xgetbv0()
{
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (uint64_t(edx) << 32) | eax;
}
#endif /* _MSC_VER */

static bool bit(uint32_t reg, int n) { return (reg >> n) & 1; }

struct Features
{
    bool avx = false;
    bool avx2 = false;
    bool avx2_vnni = false;
    bool avx512 = false;
    bool avx512_bf16 = false;
};

/* roughly what the -march targets in make-native.py require */
static Features detect_features()
{
    Features f;

    const auto max_leaf = cpuid(0).eax;
    const auto l1 = cpuid(1);

    /* OSXSAVE, AVX */
    if (!bit(l1.ecx, 27) || !bit(l1.ecx, 28))
        return f;

    /* OS must save XMM/YMM state; ZMM and opmask too for AVX-512 */
    const auto xcr0 = xgetbv0();
    if ((xcr0 & 0x6) != 0x6)
        return f;
    const bool os_avx512 = (xcr0 & 0xe6) == 0xe6;

    /* SSE4.2, POPCNT */
    f.avx = bit(l1.ecx, 20) && bit(l1.ecx, 23);
    if (!f.avx || max_leaf < 7)
        return f;

    const auto l7 = cpuid(7);
    const auto l7_1 = l7.eax >= 1 ? cpuid(7, 1) : CpuId();
    const auto ext = cpuid(0x80000000).eax >= 0x80000001 ? cpuid(0x80000001) : CpuId();

    /* AVX2, FMA, MOVBE, F16C, BMI1, BMI2, LZCNT */
    f.avx2 = bit(l7.ebx, 5) && bit(l1.ecx, 12) && bit(l1.ecx, 22) && bit(l1.ecx, 29)
        && bit(l7.ebx, 3) && bit(l7.ebx, 8) && bit(ext.ecx, 5);

    /* AVX-VNNI */
    f.avx2_vnni = f.avx2 && bit(l7_1.eax, 4);

    /* AVX512 F, DQ, CD, BW, VL */
    f.avx512 = f.avx2 && os_avx512 && bit(l7.ebx, 16) && bit(l7.ebx, 17)
        && bit(l7.ebx, 28) && bit(l7.ebx, 30) && bit(l7.ebx, 31);

    /* AVX512 VNNI, BF16 */
    f.avx512_bf16 = f.avx512 && bit(l7.ecx, 11) && bit(l7_1.eax, 5);

    return f;
}

/* best first */
static std::vector<std::string_view> supported_flavors()
{
    const auto cpu = detect_features();
    const std::pair<std::string_view, bool> flavors[] = {
        { "AVX512_BF16", cpu.avx512_bf16 },
        { "AVX512", cpu.avx512 },
        { "AVX2_VNNI", cpu.avx2_vnni },
        { "AVX2", cpu.avx2 },
        { "AVX", cpu.avx },
        { "generic", true },
    };

    std::vector<std::string_view> supported;
    for (const auto& [name, ok] : flavors)
    {
        if (ok)
            supported.push_back(name);
    }
    return supported;
}

#elif defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>

/* best first */
static std::vector<std::string_view> supported_flavors()
{
    /* -march=armv8.2-a+fp16+dotprod */
    const auto hwcap = ::getauxval(AT_HWCAP);
    if ((hwcap & HWCAP_FPHP) && (hwcap & HWCAP_ASIMDHP) && (hwcap & HWCAP_ASIMDDP))
        return { "ARMv8_2", "generic" };
    return { "generic" };
}

#else
#error "unsupported CPU architecture"
#endif /* __x86_64__ */

/* sturddle-<version> or sturddle-<version>-<suffix> */
static bool is_candidate(const fs::path& path)
{
    const auto name = program_name(path);
    return name == BASE_NAME || (name.starts_with(BASE_NAME) && name[BASE_NAME.size()] == '-');
}

static std::string read_flavor(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::string data(fs::file_size(path), '\0');
    if (!in.read(data.data(), std::streamsize(data.size())))
        throw std::runtime_error(std::format("cannot read {}", path.string()));

    const auto pos = data.find(TAG_PREFIX);
    if (pos == std::string::npos)
        return {};
    const auto start = pos + TAG_PREFIX.size();
    return data.substr(start, data.find('\0', start) - start);
}

static std::map<std::string, fs::path> find_flavors(const fs::path& dir, const fs::path& self)
{
    std::map<std::string, fs::path> found;

    for (const auto& entry : fs::directory_iterator(dir))
    {
        const auto& path = entry.path();
        if (!entry.is_regular_file() || !is_candidate(path) || fs::equivalent(path, self))
            continue;
        if (auto flavor = read_flavor(path); !flavor.empty())
            found.emplace(std::move(flavor), path);
    }
    return found;
}

struct Trailer
{
    char magic[8];
    uint64_t offset;
    uint64_t size;
};
static_assert(sizeof(Trailer) == 24);

static std::optional<Trailer> find_payload(std::istream& in, uint64_t file_size)
{
    const auto end = payload_end(in, file_size);
    if (end < sizeof(Trailer))
        return std::nullopt;

    /* the build pads the payload so that the trailer ends 8-byte aligned, as signtool wants */
    const auto trailer_pos = end - sizeof(Trailer);
    Trailer trailer;
    read_exact(in.seekg(trailer_pos), &trailer, sizeof(trailer));

    if (std::string_view(trailer.magic, sizeof(trailer.magic)) != BUNDLE_MAGIC)
        return std::nullopt;
    if (trailer.offset > trailer_pos || trailer.size > trailer_pos - trailer.offset)
        throw std::runtime_error("bundle: bad trailer");
    return trailer;
}

/* decompresses the payload on demand */
class PayloadReader
{
public:
    PayloadReader(std::istream& in, const Trailer& trailer)
        : _in(in)
        , _remaining(trailer.size)
        , _buf(ZSTD_DStreamInSize())
        , _ctx(ZSTD_createDCtx(), ZSTD_freeDCtx)
    {
        if (!_ctx)
            throw std::bad_alloc();
        _in.seekg(trailer.offset);
    }

    void read(void* data, size_t size)
    {
        ZSTD_outBuffer out = { data, size, 0 };
        while (out.pos < out.size)
        {
            if (decompress(out) == 0 && out.pos < out.size)
                throw std::runtime_error("bundle: truncated payload");
        }
    }

    template <typename T> T read()
    {
        T value;
        read(&value, sizeof(value));
        return value;
    }

    /* the checksum is verified only once the end of the frame is reached */
    void finish()
    {
        char extra;
        ZSTD_outBuffer out = { &extra, sizeof(extra), 0 };
        while (_ret != 0 && out.pos == 0)
            decompress(out);
        if (out.pos)
            throw std::runtime_error("bundle: trailing data in payload");
    }

private:
    /* returns 0 at the end of the frame */
    size_t decompress(ZSTD_outBuffer& out)
    {
        if (_input.pos == _input.size && _remaining)
        {
            const auto size = std::min<uint64_t>(_remaining, _buf.size());
            read_exact(_in, _buf.data(), size);
            _remaining -= size;
            _input = { _buf.data(), size, 0 };
        }
        const auto in_pos = _input.pos;
        const auto out_pos = out.pos;

        _ret = ZSTD_decompressStream(_ctx.get(), &out, &_input);
        if (ZSTD_isError(_ret))
            throw std::runtime_error(std::format("bundle: {}", ZSTD_getErrorName(_ret)));
        if (_ret && !_remaining && _input.pos == in_pos && out.pos == out_pos)
            throw std::runtime_error("bundle: truncated payload");
        return _ret;
    }

    std::istream& _in;
    uint64_t _remaining;
    std::vector<char> _buf;
    ZSTD_inBuffer _input = {};
    size_t _ret = 1;
    std::unique_ptr<ZSTD_DCtx, size_t (*)(ZSTD_DCtx*)> _ctx;
};

static void extract(std::istream& in, const Trailer& trailer, const fs::path& dir)
{
    PayloadReader reader(in, trailer);
    std::vector<char> buf(1 << 20);

    while (const auto name_len = reader.read<uint16_t>())
    {
        std::string name(name_len, '\0');
        reader.read(name.data(), name.size());
        if (name == "." || name == ".." || name.find_first_of("/\\:") != std::string::npos)
            throw std::runtime_error(std::format("bundle: bad file name {}", name));

        const auto exec = reader.read<uint8_t>();
        auto size = reader.read<uint64_t>();

        const auto path = dir / name;
        std::ofstream out(path, std::ios::binary);
        while (out && size)
        {
            const auto chunk = std::min<uint64_t>(size, buf.size());
            reader.read(buf.data(), chunk);
            out.write(buf.data(), std::streamsize(chunk));
            size -= chunk;
        }
        if (!out.flush())
            throw std::runtime_error(std::format("cannot write {}", path.string()));

        if (exec)
            fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add);
    }
    reader.finish();
}

class TempDir
{
public:
    TempDir()
    {
        const auto base = fs::temp_directory_path();
        do
            _path = base / std::format("{}{}-{}", TEMP_PREFIX, get_pid(), random_int(0, 999999));
        while (!create_private_dir(_path));
    }

    ~TempDir()
    {
        /* on Windows the engine exe may stay locked for a moment after it exits */
        for (int i = 0; i != 50; ++i)
        {
            std::error_code error;
            fs::remove_all(_path, error);
            if (!error)
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return _path; }

private:
    fs::path _path;
};

int main(int, char** argv)
{
    try
    {
        const auto self = self_path();
        auto dir = self.parent_path();

        std::optional<TempDir> temp_dir;
        std::ifstream in(self, std::ios::binary);

        if (const auto trailer = find_payload(in, fs::file_size(self)))
        {
            temp_dir.emplace();
            extract(in, *trailer, temp_dir->path());
            dir = temp_dir->path();
        }
        in.close();

        const auto found = find_flavors(dir, self);

        for (const auto flavor : supported_flavors())
        {
            if (const auto iter = found.find(std::string(flavor)); iter != found.end())
                return run(iter->second, argv);
        }

        throw std::runtime_error(std::format("no {} flavor for this CPU in {}", BASE_NAME, dir.string()));
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "{}", e.what());
    }
    return EXIT_FAILURE;
}
