#pragma once

#include "math.hpp"
#include "SqPatch.hpp"

#ifdef SPEC_USE_STD_MATH
#define math_namespace std
#else
#define math_namespace ::sphexa::math
#endif

namespace sphexa
{
#ifdef SPEC_USE_LT_IN_KERNELS  
template <typename T>
inline T wharmonic_lt_with_derivative(const T v, const size_t ltSize, const T* wharmonicLt, const T* wharmonicLtDer)
{
    namespace lt = sphexa::lookup_tables;

    const size_t halfTableSize = ltSize / 2.0;
    const size_t idx = v * halfTableSize;

    return (idx >= ltSize)
               ? 0.0
               : wharmonicLt[idx] + wharmonicLtDer[idx] * (v - (T)idx / halfTableSize);
}
#endif
  
template <typename T>
inline T wharmonic_derivative_std(T v)
{
    if (v == 0.0) return 0.0;

    const T Pv = (PI / 2.0) * v;
    const T sincv = std::sin(Pv) / (Pv);

    return sincv * (PI / 2.0) * ((std::cos(Pv) / std::sin(Pv)) - 1.0 / Pv);
}

  
template <typename T>
inline T wharmonic_std(T v)
{
    if (v == 0.0) return 1.0;

    const T Pv = (PI / 2.0) * v;

    return std::sin(Pv) / Pv;
}
#ifdef SPEC_USE_LT_IN_KERNELS
  constexpr auto wharmonic = wharmonic_lt_with_derivative<double>;
#else
  constexpr auto wharmonic = wharmonic_std<double>;
#endif

template <typename T>
inline T artificial_viscosity(T ro_i, T ro_j, T h_i, T h_j, T c_i, T c_j, T rv, T r_square)
{
    T alpha = 1.0;
    T beta = 2.0;
    T epsilon = 0.01;

    T ro_ij = (ro_i + ro_j) / 2.0;
    T c_ij = (c_i + c_j) / 2.0;
    T h_ij = (h_i + h_j) / 2.0;

    // calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        // calculate muij
        T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

} // namespace sphexa
