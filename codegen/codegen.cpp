/*
 * Sturddle Chess Engine (C) 2023, 2024, 2025 Cristian Vlasceanu
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

#include "hash_builder.h"

int main()
{
    HashBuilder builder;

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
