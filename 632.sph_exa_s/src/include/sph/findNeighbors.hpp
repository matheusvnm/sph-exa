#pragma once

#include <vector>
#include "Octree.hpp"
#include "LinearOctree.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template <typename T>
T normalize(T d, T min, T max)
{
    return (d - min) / (max - min);
}

#ifdef SPEC_OPENACC
#pragma acc routine
#endif
template <typename T>
void findNeighborsDispl(const size_t pi, const int *clist, const int *t_ord, const T *t_x, const T *t_y, const T *t_z, const T *x, const T *y, const T *z, const T *h,
                        const T displx, const T disply, const T displz, const size_t ngmax, int *neighbors, int *neighborsCount,
                        // The linear tree
                   		const int *o_cells, const int *o_ncells, const int *o_localPadding, const int *o_localParticleCount, const T *o_xmin, const T *o_xmax, const T *o_ymin, const T *o_ymax, const T *o_zmin, const T *o_zmax)
{
    const size_t i = clist[pi];
    int collisionsCount = 0;
    int collisionNodes[128];
    const T xi = x[i] + displx;
    const T yi = y[i] + disply;
    const T zi = z[i] + displz;
    const T ri = 2.0 * h[i];
    constexpr int nX = 2;
    constexpr int nY = 2;
    constexpr int nZ = 2;

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = -1;

    int node = 0;
    do
    {
        if (o_ncells[node] == 8)
        {
            int mix = std::max((int)(normalize(xi - ri, o_xmin[node], o_xmax[node]) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, o_ymin[node], o_ymax[node]) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, o_zmin[node], o_zmax[node]) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, o_xmin[node], o_xmax[node]) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, o_ymin[node], o_ymax[node]) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, o_zmin[node], o_zmax[node]) * nZ), nZ - 1);

            // Maximize threads sync
            for (int hz = 0; hz < 2; hz++)
            {
                for (int hy = 0; hy < 2; hy++)
                {
                    for (int hx = 0; hx < 2; hx++)
                    {
                        // if overlap
                        if (hz >= miz && hz <= maz && hy >= miy && hy <= may && hx >= mix && hx <= max)
                        {
                            const int l = hz * nX * nY + hy * nX + hx;
                            const int child = o_cells[node * 8 + l];
                            if(o_localParticleCount[child] > 0)
                                stack[stackptr++] = child;
                        }
                    }
                }
            }
        }

        if (o_ncells[node] != 8) collisionNodes[collisionsCount++] = node;

        node = stack[--stackptr]; // Pop next
    } while (node > 0);

    size_t ngc = neighborsCount[pi];
    
    for (int ni = 0; ni < collisionsCount; ni++)
    {
        int node = collisionNodes[ni];
        T r2 = ri * ri;
        for (int pj = 0; pj < o_localParticleCount[node]; pj++)
        {
            size_t j = o_localPadding[node] + pj;

            T xj = t_x[j];
            T yj = t_y[j];
            T zj = t_z[j];

            T xx = xi - xj;
            T yy = yi - yj;
            T zz = zi - zj;

            T dist = xx * xx + yy * yy + zz * zz;

            if (dist < r2 && i != j && ngc < ngmax) {
                neighbors[ngc++] = t_ord[j];
            }
        }
    }
    neighborsCount[pi] = ngc;
}

#ifdef SPEC_OPENACC
#pragma acc routine
#endif
template <typename T>
void findNeighbors(const size_t pi, const int *clist, const int *t_ord, const T *t_x, const T *t_y, const T *t_z, const T *x, const T *y, const T *z, const T *h, const T displx,
                   const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount,
                   // The linear tree
                   const int *o_cells, const int *o_ncells, const int *o_localPadding, const int *o_localParticleCount, const T *o_xmin, const T *o_xmax, const T *o_ymin, const T *o_ymax, const T *o_zmin, const T *o_zmax)
{
    T dispx[3], dispy[3], dispz[3];

    dispx[0] = 0;
    dispy[0] = 0;
    dispz[0] = 0;
    dispx[1] = -displx;
    dispy[1] = -disply;
    dispz[1] = -displz;
    dispx[2] = displx;
    dispy[2] = disply;
    dispz[2] = displz;

    neighborsCount[pi] = 0;


    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl<T>(pi, clist, t_ord, t_x, t_y, t_z, x, y, z, h, dispx[hx], dispy[hy], dispz[hz], ngmax, &neighbors[pi * ngmax], neighborsCount,
                                   // The linear tree
                                   o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
}
} // namespace kernels

template <typename T, class Dataset>
void computeFindNeighbors(const Octree<T> &tree, const LinearOctree<T> &o, const std::vector<int> &clist, Dataset &d)
{
    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;

    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

    const size_t np = d.x.size();
    const size_t ngmax = d.ngmax;

    // Device pointers
    const T *d_h = d.h.data();
    const T *d_x = d.x.data();
    const T *d_y = d.y.data();
    const T *d_z = d.z.data();
    const int TS = d.TS;
    size_t pi;

    const int *t_ord = tree.ordering->data();
    const T *t_x = tree.x->data();
    const T *t_y = tree.y->data();
    const T *t_z = tree.z->data();

    // Map LinearTree to device pointers
    // Currently OpenMP implementations do not support very well the mapping of structures
    // So we convert everything to simple arrays and pass them to OpenMP
    const size_t st = o.size;
    const size_t stt = o.size * 8;
    const int *o_ncells = o.ncells.data();
    const int *o_cells = o.cells.data();
    const int *o_localPadding = o.localPadding.data();
    const int *o_localParticleCount = o.localParticleCount.data();
    const T *o_xmin = o.xmin.data();
    const T *o_xmax = o.xmax.data();
    const T *o_ymin = o.ymin.data();
    const T *o_ymax = o.ymax.data();
    const T *o_zmin = o.zmin.data();
    const T *o_zmax = o.zmax.data();
   

    const size_t n = clist.size();
    const size_t nn = n * ngmax;
    d.resizeN(n);

    // Device pointers
    const int *d_clist = clist.data();
    int *d_neighbors = d.neighbors.data();
    int *d_neighborsCount = d.neighborsCount.data();
    int num_cores = 64;
    int blocksize2 = n/num_cores;
    

// clang-format off
#ifdef SPEC_OPENMP_TARGET
#pragma omp target teams distribute parallel for map(to: t_ord[0:np], t_x[0:np], t_y[0:np], t_z[0:np], d_x[0:np], d_y[0:np], d_z[0:np], d_h[0:np], o_cells[0:stt], o_ncells[0:st], o_localPadding[0:st], o_localParticleCount[0:st], o_xmin[0:st], o_xmax[0:st], o_ymin[0:st], o_ymax[0:st], o_zmin[0:st], o_zmax[0:st], d_clist[0:n]) map(tofrom: d_neighbors[0:nn], d_neighborsCount [0:n])
#endif
#ifdef SPEC_OPENMP
#pragma omp parallel for
#endif
#ifdef SPEC_OPENACC
#pragma acc parallel loop copyin(t_ord[0:np], t_x[0:np], t_y[0:np], t_z[0:np], d_x[0:np], d_y[0:np], d_z[0:np], d_h[0:np], o_cells[0:stt], o_ncells[0:st], o_localPadding[0:st], o_localParticleCount[0:st], o_xmin[0:st], o_xmax[0:st], o_ymin[0:st], o_ymax[0:st], o_zmin[0:st], o_zmax[0:st], d_clist[0:n]) present(d_neighbors[0:nn], d_neighborsCount [0:n])
#endif
#ifdef SPEC_OMPSS
#pragma oss taskloop grainsize(TS) out(d_neighborsCount[pi;TS])
#endif
    for (pi = 0; pi < n; pi++)
    {

        kernels::findNeighbors<T>(pi, d_clist, t_ord, t_x, t_y, t_z, d_x, d_y, d_z, d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors, d_neighborsCount,
                               // The linear tree
                               o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
    }

#pragma oss taskwait
printf("[DEBUG] Neighbors computed!\n");
}

template <typename T, class Dataset>
void findNeighbors(const Octree<T> &o, const std::vector<int> &clist, Dataset &d)
{
    LinearOctree<T> l;
    createLinearOctree(o, l);
    computeFindNeighbors<T>(o, l, clist, d);
}
} // namespace sph
} // namespace sphexa
