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
 */
/*
 * File logger for the native build (no Python).
 */
#include <chrono>
#include <ctime>
#include <format>
#include <print>

#include "common.h"
#include "utility.h"

std::atomic<LogLevel> native_log_level = LogLevel::INFO;

static std::FILE* native_log_file = stderr;

static std::tm local_time(std::time_t t)
{
    std::tm tm{};
#if _WIN32
    ::localtime_s(&tm, &t);
#else
    ::localtime_r(&t, &tm);
#endif /* _WIN32 */
    return tm;
}

void native_log_open(const std::string& path, bool separate)
{
    const auto name = separate ? std::format("{}.{}", path, get_pid()) : path;
    if (const auto file = std::fopen(name.c_str(), "a"))
        native_log_file = file;
    else
        std::println(stderr, "Cannot open {}, logging to stderr", name);
}

/* same format as the Python logger in main.py */
void native_log_message(int level, const std::string& msg, bool force)
{
    if (!force && level < int(native_log_level.load(std::memory_order_relaxed)))
        return;
    static constexpr const char* names[] = { "?", "DEBUG", "INFO", "WARNING", "ERROR" };
    static const auto pid = get_pid();
    const auto name = (level >= 1 && level <= 4) ? names[level] : names[0];
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    const auto tm = local_time(std::chrono::system_clock::to_time_t(now));

    /* no heap allocation unless the line is very long */
    char buf[4096];
    const auto end = std::format_to_n(buf, sizeof(buf), "{:04}-{:02}-{:02} {:02}:{:02}:{:02},{:03} {:<8} {} ",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ms, name, pid).out;
    const auto size = size_t(end - buf) + msg.size() + 1;

    if (size <= sizeof(buf))
    {
        std::memcpy(end, msg.data(), msg.size());
        end[msg.size()] = '\n';
        std::fwrite(buf, 1, size, native_log_file);
    }
    else
    {
        const auto line = std::string(buf, end) + msg + '\n';
        std::fwrite(line.data(), 1, line.size(), native_log_file);
    }
    std::fflush(native_log_file);
}
