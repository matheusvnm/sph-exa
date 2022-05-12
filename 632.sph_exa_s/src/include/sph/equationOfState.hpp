#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void __attribute__((noinline)) computeEquationOfState(const std::vector<int> &l, Dataset &d)
{
    const int n = l.size();
    const int *clist = l.data();
    int pi;

    const T *ro = d.ro.data();
    const T *ro_0 = d.ro_0.data();
    const T *p_0 = d.p_0.data();
    const int TS = d.TS;
    T *p = d.p.data();
    T *c = d.c.data();
    T *u = d.u.data();

    const T heatCapacityRatio = 7.0;
    const T speedOfSound0 = 3500.0;
    const T density0 = 1.0;

    // (ro_0 / 7.0) * c^2
    // const T chi = (1000.0 / 7.0) * (35.0 * 35.0);
    const T chi = (density0 / heatCapacityRatio) * (speedOfSound0 * speedOfSound0);
#if defined(SPEC_OPENMP) || defined(SPEC_OPENMP_TARGET)
#pragma omp parallel for
#endif
#ifdef SPEC_OMPSS
#pragma oss taskloop grainsize(TS) in(clist[pi;TS], ro_0[pi;TS], ro[0;n], p_0[pi;TS]) out(p[pi;TS], c[pi;TS], u[pi;TS])
#endif
    for (pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];
        p[i] = chi * (pow(ro[i] / ro_0[i], heatCapacityRatio) - 1.0) + p_0[i];
        c[i] = speedOfSound0;
        u[i] = 1.0;           
    }
    printf("[DEBUG] Equation of state computed!\n");
}
} // namespace sph
} // namespace sphexa
