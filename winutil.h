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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "ms_windows.h"
#include <tlhelp32.h>

/*
 * Windows hybrid CPU (performance / efficiency cores) detection and binding.
 */
namespace win
{
    struct Topology
    {
        bool _hybrid = false;
        std::vector<ULONG> _p_cores; /* CPU set IDs of the highest efficiency class */
        DWORD_PTR _sys_mask = 0; /* zero if the process spans multiple processor groups */

        Topology(const Topology&) = delete;
        Topology& operator=(const Topology&) = delete;

        Topology()
        {
            DWORD_PTR proc_mask = 0;
            GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &_sys_mask);

            ULONG len = 0;
            GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
            if (len == 0)
                return;

            /* uint64_t storage for alignment */
            std::vector<uint64_t> buf((len + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            const auto info = reinterpret_cast<const std::byte*>(buf.data());
            if (!GetSystemCpuSetInformation(
                    reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()), len, &len, GetCurrentProcess(), 0))
                return;

            BYTE min_class = 0xFF, max_class = 0;
            for (ULONG offset = 0; offset < len; )
            {
                const auto entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(info + offset);
                if (entry->Size == 0)
                    break;
                offset += entry->Size;
                if (entry->Type != CpuSetInformation)
                    continue;

                const auto cls = entry->CpuSet.EfficiencyClass;
                min_class = std::min(min_class, cls);
                if (cls > max_class)
                {
                    max_class = cls;
                    _p_cores.clear();
                }
                if (cls == max_class)
                    _p_cores.push_back(entry->CpuSet.Id);
            }
            _hybrid = min_class < max_class;
        }

        /** true if the process default CPU sets are exactly the P cores */
        bool process_uses_p_cores(HANDLE proc) const
        {
            ULONG ids[MAXIMUM_PROC_PER_GROUP] = {};
            ULONG count = 0;
            if (!GetProcessDefaultCpuSets(proc, ids, ULONG(std::size(ids)), &count) || count != _p_cores.size())
                return false;

            /* Windows may return the IDs in any order, so look each one up */
            return std::all_of(ids, ids + count, [this](ULONG id) {
                return std::find(_p_cores.begin(), _p_cores.end(), id) != _p_cores.end();
            });
        }
    };

    /* singleton */
    inline const Topology& topology()
    {
        static const Topology t;
        return t;
    }

    /** throw an error with the failed API name and Windows error code */
    [[noreturn]] inline void throw_last_error(std::string_view api)
    {
        const auto code = GetLastError();
        throw std::runtime_error(std::string(api) + " failed: " + std::to_string(code));
    }

    /* make calling thread follow the process defaults */
    inline void reset_thread()
    {
        const auto thread = GetCurrentThread();
        const auto sys_mask = topology()._sys_mask;

        /* if we can't read the current setting, set it anyway */
        GROUP_AFFINITY ga = {};
        if (sys_mask && (!GetThreadGroupAffinity(thread, &ga) || ga.Mask != sys_mask)
            && !SetThreadAffinityMask(thread, sys_mask))
            throw_last_error("SetThreadAffinityMask");

        /* a thread's own CPU sets override the process default;
         * with no buffer, this call just tells us how many there are */
        ULONG count = 1;
        GetThreadSelectedCpuSets(thread, nullptr, 0, &count);
        if (count && !SetThreadSelectedCpuSets(thread, nullptr, 0))
            throw_last_error("SetThreadSelectedCpuSets");
    }

    /** restrict the process to P cores and stop power throttling; throw std::runtime_error on failure */
    inline void bind_to_p_cores()
    {
        const auto proc = GetCurrentProcess();
        const auto sys_mask = topology()._sys_mask;
        DWORD_PTR proc_mask = 0, unused = 0;

        /* only change what differs; setting is slow */
        if (!GetProcessAffinityMask(proc, &proc_mask, &unused))
            throw_last_error("GetProcessAffinityMask");

        /* affinity beats CPU sets: if someone restricted us to certain cores,
         * Windows ignores our P-core choice; allow all cores first
         */
        if (proc_mask && proc_mask != sys_mask && !SetProcessAffinityMask(proc, sys_mask))
            throw_last_error("SetProcessAffinityMask");

        const auto& ids = topology()._p_cores;
        if (!topology().process_uses_p_cores(proc) && !SetProcessDefaultCpuSets(proc, ids.data(), ULONG(ids.size())))
            throw_last_error("SetProcessDefaultCpuSets");

        /* P cores alone do not help if Windows throttles the engine as a background process */
        constexpr ULONG speed = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        PROCESS_POWER_THROTTLING_STATE state = {};
        state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        if (!GetProcessInformation(proc, ProcessPowerThrottling, &state, sizeof(state))
            || !(state.ControlMask & speed) || (state.StateMask & speed))
        {
            state.ControlMask = speed;
            state.StateMask = 0; /* never throttle */
            /* fails on Windows 10 before 1709, which predates hybrid CPUs anyway */
            if (!SetProcessInformation(proc, ProcessPowerThrottling, &state, sizeof(state)))
                throw_last_error("SetProcessInformation");
        }

        reset_thread();
    }

    /** let Windows schedule and throttle the process as it sees fit; throw std::runtime_error on failure */
    inline void unbind()
    {
        const auto proc = GetCurrentProcess();
        if (!SetProcessDefaultCpuSets(proc, nullptr, 0))
            throw_last_error("SetProcessDefaultCpuSets");

        /* let Windows decide about throttling again */
        PROCESS_POWER_THROTTLING_STATE state = {};
        state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        if (!SetProcessInformation(proc, ProcessPowerThrottling, &state, sizeof(state)))
            throw_last_error("SetProcessInformation");
    }

    /*
     * Helpers for manage_console (see below).
     */
    inline DWORD get_parent_pid(DWORD processId)
    {
        HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (hSnapshot == INVALID_HANDLE_VALUE)
            return 0;

        std::unique_ptr<void, decltype(&CloseHandle)> cleanup(hSnapshot, &CloseHandle);

        PROCESSENTRY32 pe32 = {};
        pe32.dwSize = sizeof(PROCESSENTRY32);

        if (!Process32First(hSnapshot, &pe32))
            return 0;

        do {
            if (pe32.th32ProcessID == processId)
            {
                // Running as a python script? bail
                if (_strnicmp(pe32.szExeFile, "python", 6) == 0)
                    return 0;

                return pe32.th32ParentProcessID;
            }
        } while (Process32Next(hSnapshot, &pe32));

        return 0;
    }

    /** return true if a console was allocated; throw std::runtime_error on failure */
    inline bool ensure_console()
    {
        if (GetConsoleWindow())
            return false;

        if (!AllocConsole())
            throw_last_error("AllocConsole");

        // Rebind standard handles
        FILE* fp = nullptr;
        freopen_s(&fp, "CONIN$",  "r", stdin);
        freopen_s(&fp, "CONOUT$", "w", stdout);
        freopen_s(&fp, "CONOUT$", "w", stderr);

        return true;
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
     * Return true if a console was allocated; throw std::runtime_error if allocation failed.
     */
    inline bool manage_console()
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
} /* namespace win */
