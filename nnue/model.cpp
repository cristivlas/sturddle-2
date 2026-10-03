/*
 * Sturddle Chess Engine (C) 2023 - 2026 Cristian Vlasceanu
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
#include <filesystem>
#include <fstream>
#include "model.h"

using namespace nnue;

#if SHARED_WEIGHTS

void Model::init()
{
    if (!default_weights_path.empty())
    {
        load_weights(default_weights_path);
    }
}

#else

/* Use C23/C++26 #embed of weights.bin */

#if !defined(__has_embed)
  #error "embedded build requires #embed support (GCC 15+, Clang 19+, MSVC 17.15+)"
#endif
#if __has_embed("weights.bin") != __STDC_EMBED_FOUND__
  #error "weights.bin not found; run tools/fetch_weights.py before building"
#endif

#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wc23-extensions"
#elif defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"       /* umbrella: any gcc with #embed */
  #pragma GCC diagnostic ignored "-Wc++26-extensions" /* precise name where recognized */
#endif

void Model::init()
{
    static constexpr unsigned char WEIGHTS_DATA[] = {
        #embed "weights.bin"
    };

    struct membuf : std::streambuf
    {
        membuf(const char* data, size_t size)
        {
            char* p = const_cast<char*>(data);
            setg(p, p, p + size);
        }
    } buf(reinterpret_cast<const char*>(WEIGHTS_DATA), sizeof(WEIGHTS_DATA));

    std::istream file(&buf);
    file.exceptions(std::ios::failbit | std::ios::badbit);

    /* Same order as Model::load_weights file-based path */
    L1A.load_weights(file);
    L1B.load_weights(file);
#if NNUE_HIDDEN_1C
    L1C.load_weights(file);
    fuse_modulation();
#endif /* NNUE_HIDDEN_1C */
    POOL.load_weights(file);
    L2.load_weights(file);
    L3.load_weights(file);
    EVAL.load_weights(file);
#if USE_MOVE_PREDICTION
    LMOVE_ACC.load_weights(file);
    LMOVES.load_weights(file);
#endif
}

#if defined(__clang__)
  #pragma clang diagnostic pop
#elif defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#endif /* !SHARED_WEIGHTS */


void Model::validate_weights_file(const std::filesystem::path& weights_path)
{
    constexpr auto param_count =
        L1AType::param_count()
        + L1BType::param_count()
    #if NNUE_HIDDEN_1C
        + L1CType::param_count()
    #endif /* NNUE_HIDDEN_1C */
        + PoolType::param_count()
        + L2Type::param_count()
        + L3Type::param_count()
        + EVALType::param_count()
    #if USE_MOVE_PREDICTION
        + LMOVEAccType::param_count()
        + LMOVEType::param_count()
    #endif
        ;
    constexpr auto expected_size = param_count * sizeof(float);
    const auto file_size = std::filesystem::file_size(weights_path);
    if (file_size != expected_size)
        throw std::runtime_error(weights_path.string() + ": expected " + std::to_string(expected_size) + " bytes, got " + std::to_string(file_size));
}


void Model::load_weights(const std::filesystem::path& weights_path)
{
    validate_weights_file(weights_path);

    std::ifstream file(weights_path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open weights file: " + weights_path.string());

    file.exceptions(std::ios::failbit | std::ios::badbit);

    try
    {
        /* Load layers in the same order that the trainer exports them. */
        L1A.load_weights(file);
        L1B.load_weights(file);
    #if NNUE_HIDDEN_1C
        L1C.load_weights(file);
    #endif /* NNUE_HIDDEN_1C */
        POOL.load_weights(file);
        L2.load_weights(file);
        L3.load_weights(file);
        EVAL.load_weights(file);

    #if USE_MOVE_PREDICTION
        LMOVE_ACC.load_weights(file);
        LMOVES.load_weights(file);
    #endif
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("Error reading weights from: " + weights_path.string());
    }
#if NNUE_HIDDEN_1C
    fuse_modulation();
#endif /* NNUE_HIDDEN_1C */
}
