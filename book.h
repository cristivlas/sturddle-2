#pragma once
#if _WIN32
 #include "ms_windows.h"
#else
  #include <sys/mman.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif /* !_WIN32 */
#include <string>
#include <vector>


class MemoryMappedFile
{
public:
    MemoryMappedFile() = default;
    ~MemoryMappedFile() { close(); }

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    void close()
    {
        if (_data)
        {
    #ifdef _WIN32
            UnmapViewOfFile(_data);
    #else
            munmap(const_cast<char*>(_data), _size);
    #endif
        }
        _data = nullptr;
        _size = 0;
    }

    bool open(const std::string& filename)
    {
        close();

    #ifdef _WIN32
        HANDLE hFile = CreateFileA(
            filename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (hFile == INVALID_HANDLE_VALUE)
            return false;

        _size = GetFileSize(hFile, nullptr);
        HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap)
        {
            CloseHandle(hFile);
            return false;
        }

        _data = static_cast<const char*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, _size));
        CloseHandle(hFile);
        CloseHandle(hMap);
    #else
        int fd = ::open(filename.c_str(), O_RDONLY);
        if (fd < 0)
            return false;

        _size = lseek(fd, 0, SEEK_END);
        _data = static_cast<const char*>(mmap(nullptr, _size, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
    #endif /* !_WIN32 */
        return true;
    }

    const char* data() const { return _data; }
    size_t size() const { return _size; }

private:
    const char* _data = nullptr;
    size_t _size = 0;
};


struct PolyglotEntry
{
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};


#if _MSC_VER
INLINE uint64_t swap_uint64(uint64_t val) { return _byteswap_uint64(val); }
INLINE uint16_t swap_uint16(uint16_t val) { return _byteswap_ushort(val); }
#else
INLINE uint64_t swap_uint64(uint64_t val) { return __builtin_bswap64(val); }
INLINE uint16_t swap_uint16(uint16_t val) { return __builtin_bswap16(val); }
#endif /* _MSC_VER */


/* Polyglot opening book */
class PolyglotBook
{
public:
    enum LookupMode { FIRST_MATCH, BEST_WEIGHT, WEIGHTED_CHOICE };

    bool open(const std::string& filename) { return _mapped_file.open(filename); }
    void close() { _mapped_file.close(); }

    bool is_open() const { return _mapped_file.size(); }

    uint16_t lookup_move(uint64_t key, LookupMode mode)
    {
        const auto& entries = find_moves(key);
        if (entries.empty())
            return 0;

        if (mode == FIRST_MATCH)
        {
            return entries[0].move;
        }
        else if (mode == BEST_WEIGHT)
        {
            return std::max_element(entries.begin(), entries.end(), [](const PolyglotEntry& a, const PolyglotEntry& b) {
                return a.weight < b.weight;
            })->move;
        }
        else if (mode == WEIGHTED_CHOICE)
        {
            int totalWeight = 0;
            for (const auto& entry : entries) totalWeight += entry.weight;

            const auto choice = random_int(0, totalWeight);

            int currentSum = 0;
            for (const auto& entry : entries)
            {
                currentSum += entry.weight;
                if (currentSum > choice)
                    return entry.move;
            }
            ASSERT(false);
        }

        return 0;
    }

private:
    // Find all moves for a given position key using binary search
    const std::vector<PolyglotEntry>& find_moves(uint64_t key)
    {
        const auto size = _mapped_file.size() / sizeof(PolyglotEntry);
        const auto begin = reinterpret_cast<const PolyglotEntry*>(_mapped_file.data());
        const PolyglotEntry* end = begin + size;

        const PolyglotEntry* it = std::lower_bound(begin, end, key, [](const PolyglotEntry& entry, uint64_t key) {
            return swap_uint64(entry.key) < key;
        });

        _entries.clear();

        // Collect all matching entries
        while (it != end && swap_uint64(it->key) == key)
        {
            PolyglotEntry entry = *it++;
            entry.key = key;
            entry.move = swap_uint16(entry.move);
            entry.weight = swap_uint16(entry.weight);
            _entries.emplace_back(entry);
        }

        return _entries;
    }

    MemoryMappedFile _mapped_file;
    std::vector<PolyglotEntry> _entries;
};
