/*
 * Sturddle Chess Engine (C) 2023, 2024 Cristian Vlasceanu
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

#include <array>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>
#include "chess.h"

using namespace chess;

using HashTable = std::unordered_map<Bitboard, Bitboard>;
using PerfectHash = std::map<uint16_t, Bitboard>;

struct HashParam
{
    uint64_t mask;
    uint64_t mul;
    uint16_t shift;
    PerfectHash table;
};

template <AttacksType> struct TableSize
{
    static constexpr size_t value = 1UL << 10;
};

template <> struct TableSize<AttacksType::File>
{
    static constexpr size_t value = 1UL << 6;
};

template <> struct TableSize<AttacksType::Rank>
{
    static constexpr size_t value = 1UL << 6;
};

template <> struct TableSize<AttacksType::Rook>
{
    static constexpr size_t value = 1UL << 13;
};

template <AttacksType T> struct Offset
{
    static constexpr AttacksType prev_type = AttacksType(static_cast<int>(T) - 1);
    static constexpr size_t value = Offset<prev_type>::value + TableSize<prev_type>::value * 64;
};

template <> struct Offset<AttacksType::Diag>
{
    static constexpr size_t value = 0;
};

struct Group
{
    const size_t _index;
    const size_t _offset;
    const size_t _table_size;

    std::array<HashParam, 64> _hash_info;

    Group(size_t index, size_t offset, size_t table_size)
        : _index(index)
        , _offset(offset)
        , _table_size(table_size)
    {}

    void set_hash_info(Square square, uint64_t mask, uint64_t mul, uint16_t shift, const PerfectHash& table)
    {
        ASSERT_ALWAYS(square != Square::UNDEFINED);
        _hash_info[square].mask = mask;
        _hash_info[square].mul = mul;
        _hash_info[square].shift = shift;
        _hash_info[square].table = table;
    #if DEBUG
        std::clog << square << ": mul=" << std::hex << mul << std::dec
                  << ", shift=" << shift << ", " << table.size() << " entries\n";
    #endif /* DEBUG */
    }
};

uint64_t random_uint64()
{
    // Create a static random device
    static std::random_device rd;

    // Use the device to seed a static Mersenne twister engine
    static std::mt19937_64 engine(rd());

    // Create a static uniform distribution from 0 to UINT64_MAX
    static std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    // Generate and return a random number
    return dist(engine) | dist(engine) | dist(engine) | dist(engine);
}

class HashBuilder
{
    std::vector<Group> _groups;

public:
    template <AttacksType type>
    bool build_group(const std::vector<int>& deltas)
    {
        Group group(_groups.size(), Offset<type>::value, TableSize<type>::value);

        // Caller must build groups in type order.
        ASSERT_ALWAYS(size_t(type) == group._index);

        for (int square = 0; square < 64; ++square)
        {
            HashTable attacks;

            const auto mask = sliding_attacks(square, 0, deltas) & ~edges(square);
            for_each_subset(mask, [&](Bitboard subset) {
                attacks.emplace(subset, sliding_attacks(square, subset, deltas));
            });

            if (!build<type>(Square(square), mask, attacks, group))
            {
                std::cerr << "Could not build hash for: " << square << std::endl;
                return false;
            }
        }
        _groups.emplace_back(group);
        return true;
    }

    void write(std::ostream& out) const
    {
        ASSERT_ALWAYS(_groups.size() == 4);

        constexpr size_t total_table_size =
            TableSize<AttacksType::Diag>::value * 64 +
            TableSize<AttacksType::File>::value * 64 +
            TableSize<AttacksType::Rank>::value * 64 +
            TableSize<AttacksType::Rook>::value * 64;

        out << "#pragma once\n";
        out << "/*\n";
        out << " * Auto-generated attack tables (see codegen/).\n";
        out << " * https://www.chessprogramming.org/Magic_Bitboards\n";
        out << " */\n";
    #if LOCK_TABLE_DATA
        out << "#if defined(__linux__) || defined(__APPLE__)\n";
        out << "    #include <sys/mman.h>\n";
        out << "#endif\n";
    #endif /* LOCK_TABLE_DATA */
        out << "#include <cstdint>\n";
        out << "#include <cstring>\n\n";
        out << "namespace chess {\n";
        out << "constexpr struct HashInfo {\n";
        out << "    uint64_t mask;\n";
        out << "    uint64_t mul;\n";
        out << "    uint32_t shift;\n";
        out << "} hash_info[" << _groups.size() * 64 << "] = {\n";
        for (const auto& group : _groups)
            for (const auto& h : group._hash_info)
                out << "    { 0x" << std::hex << h.mask << ", 0x"
                    << h.mul << std::dec << ", " << h.shift << " },\n";
        out << "};\n\n";
        out << "template <int> struct GroupInfo;\n\n";
        for (const auto& group : _groups)
        {
            out << "template <> struct GroupInfo<" << group._index << "> {\n";
            out << "    static constexpr size_t offset = " << group._offset << "UL;\n";
            out << "    static constexpr size_t table_size = " << group._table_size << "UL;\n";
            out << "};\n";
        };
        out << "\n";
        out << "/********************************************************\n";
        out << " *\n";
        out << " ********************************************************/\n";
        out << "struct AttackTable {\n";
        out << "    alignas(64) uint64_t _data[" << total_table_size << "];\n\n";
        out << "#ifdef DEFINE_ATTACK_TABLE_CTOR\n";
        out << "    AttackTable() {\n";
        out << "        memset(_data, 0, sizeof(_data));\n";
        for (const auto& group : _groups)
        {
            for (size_t i = 0; i < 64; ++i) // iterate over squares
            {
                for (const auto& hash : group._hash_info[i].table)
                {
                    out << "        _data[";
                    out << group._offset + i * group._table_size + hash.first << "] = 0x";
                    out << std::hex << hash.second << std::dec << ";\n";
                }
            }
        }
    #if LOCK_TABLE_DATA
        out << "    #ifdef _WIN32\n";
        out << "        VirtualLock(_data, sizeof(_data));\n";
        out << "    #elif defined(__linux__) || defined(__APPLE__)\n";
        out << "        mlock(_data, sizeof(_data));\n";
        out << "    #endif\n";
    #endif /* LOCK_TABLE_DATA */
        out << "    }\n\n";
        out << "#else\n";
        out << "    AttackTable();\n";
        out << "#endif /* DEFINE_ATTACK_TABLE_CTOR */\n\n";
        out << "    template <int Group>\n";
        out << "    INLINE size_t hash(unsigned square, uint64_t occupancy_mask) const\n";
        out << "    {\n";
        out << "        const auto& hi = hash_info[Group * 64 + square];\n";
        out << "        occupancy_mask &= hi.mask;\n";
        out << "        occupancy_mask *= hi.mul;\n";
        out << "        occupancy_mask >>= hi.shift;\n";
        out << "        occupancy_mask &= GroupInfo<Group>::table_size - 1;\n";
        out << "        return GroupInfo<Group>::offset + square * GroupInfo<Group>::table_size + occupancy_mask;\n";
        out << "    }\n\n";
        out << "    template <int Group>\n";
        out << "    INLINE uint64_t get(int square, uint64_t occupancy_mask) const\n";
        out << "    {\n";
        out << "        return _data[hash<Group>(square, occupancy_mask)];\n";
        out << "    }\n";
        out << "};\n\n";
        out << "extern const AttackTable attack_table;\n";

        static const std::unordered_map<size_t, std::string> type = {
            { size_t(AttacksType::Diag), "AttacksType::Diag" },
            { size_t(AttacksType::File), "AttacksType::File" },
            { size_t(AttacksType::Rank), "AttacksType::Rank" },
            { size_t(AttacksType::Rook), "AttacksType::Rook" },
        };

        for (const auto& group : _groups)
        {
            out << "\ntemplate<> struct Attacks<" << type.at(group._index) << ">\n";
            out << "{\n";
            out << "    INLINE static uint64_t get(int square, uint64_t mask)\n";
            out << "    {\n";
            out << "        return attack_table.get<" << group._index << ">(square, mask);\n";
            out << "    }\n";
            out << "};\n";
        }
        out << "} /* namespace chess */\n";
    }

private:
    template <AttacksType type>
    bool build(Square square, Bitboard mask, const HashTable& attacks, Group& group)
    {
        PerfectHash table;

        for (size_t n = 0; n != 10000; ++n)
        {
            const uint64_t mul = random_uint64();
            for (uint16_t shift = 1; shift < 64; ++shift)
            {
                if (try_hash<type>(mask, attacks, mul, shift, table))
                {
                    group.set_hash_info(square, mask, mul, shift, table);
                    return true;
                }
            }
        }
        return false;
    }

    template <AttacksType type>
    bool try_hash(Bitboard mask, const HashTable& attacks, uint64_t mul, uint16_t shift, PerfectHash& table)
    {
        auto hash = [mask, mul, shift](Bitboard val) -> size_t {
            return (((mask & val) * mul) >> shift) & (TableSize<type>::value - 1);
        };
        table.clear();

        for (const auto& elem : attacks)
        {
            const auto hval = hash(elem.first);
            auto iter = table.find(hval);
            if (iter == table.end())
                table.emplace(hval, elem.second);
            else if (iter->second != elem.second)
                return false; // collision!
        }

        for (const auto& elem : attacks)
            ASSERT_ALWAYS(elem.second == table[hash(elem.first)]);

        return true;
    }
};
