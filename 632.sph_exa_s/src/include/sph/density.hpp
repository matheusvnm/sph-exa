#pragma once

#include <cmath>
#include <vector>
#include "utils.hpp"

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeDensityImpl(int it, int partition_size, const std::vector<int> &l, Dataset &d)
{
    size_t pi;
    const long int n = partition_size;
    const size_t ngmax = d.ngmax;
    const size_t nOffset = it*partition_size;
    const size_t neighborsOffset = it*partition_size*ngmax;

    const int TS = d.TS;
    const int *clist = &l[nOffset];
    const int *neighbors = &d.neighbors[neighborsOffset];
    const int *neighborsCount = &d.neighborsCount[nOffset];
    T *ro = &d.ro[nOffset];

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    const T *whLt = d.wharmonicLookupTable;
    const T *whDerLt = d.wharmonicDerivativeLookupTable;
    const size_t whSize = d.wharmonicLookupTableSize;

    const BBox<T> bbox = d.bbox;

    const T K = d.K;
    const T sincIndex = d.sincIndex;

#ifdef SPEC_OPENMP_TARGET
    const int np = d.x.size();
    const int64_t allNeighbors = n * ngmax;
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeDensityImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);

    #ifdef SPEC_USE_LT_IN_KERNELS
    #pragma omp target teams distribute parallel for map(to                                                                                \
                             : clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np],      \
                             whLt [0:whSize], whDerLt [0:whSize],                                                                          \
                             z [0:np]) map(from                                                                                            \
                                             : ro [0:n])
    #else
    #pragma omp target teams distribute parallel for map(to                                                                                \
                             : clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np],      \
                             z [0:np]) map(from                                                                                            \
                                             : ro [0:n])
    #endif
#elif defined(SPEC_OPENACC)
    const int np = d.x.size();
    const int64_t allNeighbors = n * ngmax;
    #ifdef SPEC_USE_LT_IN_KERNELS
    #pragma acc parallel loop copyin(clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                            whLt [0:whSize], whDerLt [0:whSize],                                                                            \
                            z[0:np], bbox) copyout(ro [0:n]) default(present)
    #else
    #pragma acc parallel loop copyin(clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                            z[0:np], bbox) copyout(ro [0:n]) default(present)
    #endif
#else
#ifdef SPEC_OPENMP
#pragma omp parallel for
#endif
#endif
#ifdef SPEC_OMPSS
#pragma oss task inout(*ro) wait 
{
#pragma oss taskloop grainsize(TS)
#endif
    for (pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T roloc = 0.0;

        // int converstion to avoid a bug that prevents vectorization with some compilers
        // The value of NN defines the blocking of which the neighbors will be acessed [can be from 0 to ngmax].
        for (int pj = 0; pj < nn; pj++)
        {
            const int j = neighbors[pi * ngmax + pj];

            // later can be stores into an array per particle
            T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor

            // calculate the v as ratio between the distance and the smoothing length
            T vloc = dist / h[i];

#ifndef NDEBUG
            if (vloc > 2.0 + 1e-6 || vloc < 0.0)
                printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, x[i], y[i], z[i], x[j],
                       y[j], z[j], dist, h[i]);
#endif

	    #if defined(SPEC_USE_LT_IN_KERNELS)
            const T w = K * math_namespace::pow(wharmonic(vloc, whSize, whLt, whDerLt), (int)sincIndex);
	    #else
	    const T w = K * math_namespace::pow(wharmonic(vloc), (int)sincIndex);
	    #endif
	    
            const T value = w / (h[i] * h[i] * h[i]);
            roloc += value * m[j];
        }

        ro[pi] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
#ifndef NDEBUG
        if (std::isnan(ro[i])) printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
#endif
    }
#ifdef SPEC_OMPSS
}
#endif 
}

template <typename T, class Dataset>
void computeDensity(const std::vector<int> &l, Dataset &d)
{
    int n = l.size();
    int partition_size = n / d.noOfGpuLoopSplits;
#if defined(SPEC_OPENMP_TARGET) || defined(SPEC_OPENACC)
    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    const T *whLt = d.wharmonicLookupTable;
    const T *whDerLt = d.wharmonicDerivativeLookupTable;
    const size_t whSize = d.wharmonicLookupTableSize;
    const int np = d.x.size();

# ifdef SPEC_OPENMP_TARGET
    #ifdef SPEC_USE_LT_IN_KERNELS
    #pragma omp target data map(to: m [0:np], h [0:np], x [0:np], y [0:np],    \
                                    whLt [0:whSize], whDerLt [0:whSize],       \
                                    z [0:np])
    #else
    #pragma omp target data map(to: m [0:np], h [0:np], x [0:np], y [0:np],    \
                                    z [0:np])
    #endif
# else
    #ifdef SPEC_USE_LT_IN_KERNELS
    #pragma acc data copyin(m[0:np], h[0:np], x[0:np], y[0:np],    \
                            whLt[0:whSize], whDerLt[0:whSize],     \
                            z[0:np])
    #else
    #pragma acc data copyin(m[0:np], h[0:np], x[0:np], y[0:np],    \
                            z[0:np])
    #endif
# endif
#endif // SPEC_OPENMP_TARGET && SPEC_OPENACC
    for (int it = 0; it < d.noOfGpuLoopSplits; it++)
    {
        computeDensityImpl<T>(it, partition_size, l, d);
    }
    printf("[DEBUG] Density!\n");
}

template <typename T, class Dataset>
void initFluidDensityAtRest(const std::vector<int> &l, Dataset &d)
{
    size_t pi;
    const T *ro = d.ro.data();
    const int TS = d.TS;
    const int n = l.size();
    const int *clist = l.data();
    T *ro_0 = d.ro_0.data();

#if defined(SPEC_OPENMP) || defined(SPEC_OPENMP_TARGET)
#pragma omp parallel for
#endif
#ifdef SPEC_OMPSS
#pragma oss taskloop grainsize(TS) in(clist[pi;TS], ro[0;n]) out(ro_0[pi;TS])
#endif
    for (pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        ro_0[i] = ro[i];
    }
    printf("[DEBUG] Density at rest computed!\n");
}
} // namespace sph
} // namespace sphexa
