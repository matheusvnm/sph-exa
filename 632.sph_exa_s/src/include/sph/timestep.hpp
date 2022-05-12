#pragma once

#include <vector>
#include <math.h>
#include <algorithm>

#include "kernels.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeTimestep(const std::vector<int> &l, Dataset &d)
{
    int pi;
    const int n = l.size();
    const int *clist = l.data();
    int rank = d.rank;

    const T *h = d.h.data();
    const T *c = d.c.data();
    const T *dt_m1 = d.dt_m1.data();
    const T Kcour = d.Kcour;
    const T maxDtIncrease = d.maxDtIncrease;
    const int TS = d.TS;

    T &ttot = d.ttot;
    T *dt = d.dt.data();

    T mini = INFINITY;

#if defined(SPEC_OPENMP) || defined(SPEC_OPENMP_TARGET)
#pragma omp parallel for reduction(min : mini)
#endif
#ifdef SPEC_OMPSS
#pragma oss taskloop grainsize(TS) in(clist[pi;TS], h[pi;TS], c[pi;TS]) inout(dt[pi;TS]) reduction(min: mini)
#endif
    for (pi = 0; pi < n; pi++)
    {
        int i = clist[pi];
        dt[i] = Kcour * (h[i] / c[i]);
        if (dt[i] < mini) mini = dt[i];
    }
printf("[DEBUG] Timestep PT-1 computed!\n");

#ifdef SPEC_OMPSS
#pragma oss taskwait
#endif   
    if (n > 0) mini = std::min(mini, maxDtIncrease * dt_m1[0]);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &mini, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

#if defined(SPEC_OPENMP) || defined(SPEC_OPENMP_TARGET)
#pragma omp parallel for
#endif
#ifdef SPEC_OMPSS
#pragma oss taskloop grainsize(TS) in(clist[pi;TS], mini) out(dt[pi;TS]) 
#endif
    for (pi = 0; pi < n; pi++)
    {
        int i = clist[pi];
        dt[i] = mini;
    }   
      
#ifdef SPEC_OMPSS
#pragma oss taskwait
#endif
ttot += mini;
printf("[DEBUG] Timestep PT-2 computed!\n");


}
} // namespace sph
} // namespace sphexa
