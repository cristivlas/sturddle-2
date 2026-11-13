/*
 * Launcher for the native build: picks the best flavor for this CPU
 * among the sturddle-<version>* executables in its own directory,
 * identified by the STURDDLE_FLAVOR= tag that each one carries.
 */
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common.h"
#include "utility.h"
#include "version.h"

namespace fs = std::filesystem;

static constexpr std::string_view TAG_PREFIX = "STURDDLE_FLAVOR=";
static constexpr std::string_view BASE_NAME = "sturddle-" STURDDLE_VERSION;

struct CpuId
{
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

#if _WIN32
#if !defined(INFINITE)
  constexpr DWORD INFINITE = 0xFFFFFFFF;
#endif

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

    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = ::GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = ::GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = ::GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi = {};
    OS_CALL(CreateProcessW,
        target.c_str(),
        cmd.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_SUSPENDED,
        nullptr,
        nullptr,
        &si,
        &pi);
    try
    {
        OS_CALL(AssignProcessToJobObject, job, pi.hProcess);
    }
    catch (...)
    {
        /* not in the job, so it would outlive us suspended */
        ::TerminateProcess(pi.hProcess, EXIT_FAILURE);
        throw;
    }
    OS_CALL_CHECK(DwordResult, ResumeThread, pi.hThread);
    ::CloseHandle(pi.hThread);

    /* ignore Ctrl-C, leave it to the engine; after CreateProcess so it is not inherited */
    OS_CALL(SetConsoleCtrlHandler, nullptr, TRUE);

    OS_CALL_CHECK(DwordResult, WaitForSingleObject, pi.hProcess, INFINITE);

    DWORD code = EXIT_FAILURE;
    OS_CALL(GetExitCodeProcess, pi.hProcess, &code);
    return int(code);
}
#else
#include <unistd.h>

static fs::path self_path()
{
    return fs::read_symlink("/proc/self/exe");
}

static std::string program_name(const fs::path& path)
{
    return path.filename().string();
}

static int run(const fs::path& target, char** argv)
{
    const auto path = target.string();
    std::vector<char*> args = { const_cast<char*>(path.c_str()) };
    for (auto arg = argv + 1; *arg; ++arg)
        args.push_back(*arg);
    args.push_back(nullptr);

    return OS_CALL_CHECK(PosixResult, execv, path.c_str(), args.data());
}
#endif /* _WIN32 */


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

static std::map<std::string, fs::path> find_flavors(const fs::path& self)
{
    std::map<std::string, fs::path> found;

    for (const auto& entry : fs::directory_iterator(self.parent_path()))
    {
        const auto& path = entry.path();
        if (!entry.is_regular_file() || !is_candidate(path) || fs::equivalent(path, self))
            continue;
        if (auto flavor = read_flavor(path); !flavor.empty())
            found.emplace(std::move(flavor), path);
    }
    return found;
}

int main(int, char** argv)
{
    try
    {
        const auto self = self_path();
        const auto found = find_flavors(self);
        const auto cpu = detect_features();

        /* best first */
        const std::pair<std::string_view, bool> flavors[] = {
            { "AVX512_BF16", cpu.avx512_bf16 },
            { "AVX512", cpu.avx512 },
            { "AVX2_VNNI", cpu.avx2_vnni },
            { "AVX2", cpu.avx2 },
            { "AVX", cpu.avx },
        };

        for (const auto& [name, supported] : flavors)
        {
            if (!supported)
                continue;
            if (const auto iter = found.find(std::string(name)); iter != found.end())
                return run(iter->second, argv);
        }

        throw std::runtime_error(std::format("no {} flavor for this CPU in {}", BASE_NAME, self.parent_path().string()));
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "{}", e.what());
    }
    return EXIT_FAILURE;
}
