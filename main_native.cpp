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
 * Native executable entry point. Parses a minimal set of command-line
 * options and hands off to the UCI loop defined in uci_native.cpp.
 */
#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "common.h"
#include "context.h"
#include "version.h"

namespace fs = std::filesystem;

static constexpr const char* ENGINE_NAME = "Sturddle";
static constexpr const char* ENGINE_VERSION = STURDDLE_VERSION;

static void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -D, --dev-mode    enable developer-mode UCI options\n"
        "  -v, --verbose     enable verbose (debug) logging\n"
        "  -h, --help        show this help\n",
        argv0);
}

int main(int argc, char** argv)
{
    bool debug = false;
    bool dev_mode = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-D" || arg == "--dev-mode")
            dev_mode = true;
        else if (arg == "-v" || arg == "--verbose")
            debug = true;
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    const auto exe_dir = fs::absolute(fs::path(argv[0])).parent_path().string();

    try
    {
        search::Context::init(exe_dir);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 2;
    }

    std::unordered_map<std::string, std::string> params;
    params["name"] = ENGINE_NAME;
    params["version"] = std::string(ENGINE_VERSION) + "." + timestamp();
    params["dir"] = exe_dir;
    if (debug)
        params["debug"] = "true";
    if (dev_mode)
        params["dev_mode"] = "true";

    uci_loop(std::move(params));
    return 0;
}
