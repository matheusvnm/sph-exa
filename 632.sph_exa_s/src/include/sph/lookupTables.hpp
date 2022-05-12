#pragma once

#include <array>
#include "../math.hpp"
#include "sphUtils.hpp"

namespace sphexa
{

template <typename T>
inline T wharmonic_std(T v);

template <typename T>
inline T wharmonic_derivative_std(T v);

namespace lookup_tables
{
template <typename T, std::size_t N>
std::array<T, N> createWharmonicLookupTable()
{
    std::array<T, N> lt;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        lt[i] = wharmonic_std(normalizedVal);
    }
    return lt;
}

template <typename T, std::size_t N>
std::array<T, N> createWharmonicDerivativeLookupTable()
{
    std::array<T, N> lt;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        lt[i] = wharmonic_derivative_std(normalizedVal);
    }

    return lt;
}

} // namespace lookup_tables
} // namespace sphexa
