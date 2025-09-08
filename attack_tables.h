#pragma once
/*
 * Sturddle Chess Engine (C) 2022, 2023, 2024 Cristian Vlasceanu
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

namespace chess
{
    enum class AttacksType : int
    {
        Diag,
        Rook,
        File,
        Rank,
    };

    template<AttacksType> struct Attacks {};

} /* namespace chess */


/*
 https://github.com/llvm/llvm-project/issues/55798
 workaround: turn on USE_MAGIC_BITS on ARM
*/
#if USE_MAGIC_BITS && __ARM__
  #if defined(__clang__)
      #pragma clang optimize off
  #elif defined(__GNUC__)
      #pragma GCC push_options
      #pragma GCC optimize("O0")
  #endif
#endif

/* Attack tables are needed for initializing Rays */
#if TESTGEN
  #include "codegen/test.h"
#else
  #include "attacks.h"
#endif

#if USE_MAGIC_BITS && __ARM__
  #if defined(__clang__)
      #pragma clang optimize on
  #elif defined(__GNUC__)
      #pragma GCC pop_options
  #endif
#endif


namespace chess
{
    static Attacks<AttacksType::Diag> BB_DIAG_ATTACKS;
    static Attacks<AttacksType::File> BB_FILE_ATTACKS;
    static Attacks<AttacksType::Rank> BB_RANK_ATTACKS;
    static Attacks<AttacksType::Rook> BB_ROOK_ATTACKS;

} /* namespace chess */
