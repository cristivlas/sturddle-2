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
/* Generates the attacks.h file */

#include <charconv>
#include <string_view>
#include "hash_builder.h"

static std::optional<uint64_t> parse_seed(std::string_view arg)
{
    uint64_t seed = 0;
    const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), seed);
    if (ec != std::errc() || end != arg.data() + arg.size())
        return std::nullopt;
    return seed;
}

int main(int argc, char* argv[])
{
    std::optional<uint64_t> seed;

    if (argc > 1)
    {
        if (argc == 3)
        {
            seed = parse_seed(argv[2]);
        }
        else if (argc == 2)
        {
            std::random_device rd;
            seed = (uint64_t(rd()) << 32) | rd();
        }
        if (std::string_view(argv[1]) != "--search" || !seed)
        {
            std::cerr << "usage: " << argv[0] << " [--search [seed]]\n";
            return -1;
        }
        std::cerr << "seed: " << *seed << "\n";
    }

    HashBuilder builder(seed);

    chess::_init();

    if (builder.build_group<AttacksType::Diag>({-9, -7, 7, 9})
     && builder.build_group<AttacksType::Rook>({-8, -1, 1, 8})
     && builder.build_group<AttacksType::File>({-8, 8})
     && builder.build_group<AttacksType::Rank>({-1, 1}))
    {
        builder.write(std::cout);
        return 0;
    }
    return -1;
}
