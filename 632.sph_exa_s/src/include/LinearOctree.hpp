#pragma once

#include <vector>
#include "Octree.hpp"

namespace sphexa
{
template <typename T>
struct LinearOctree
{
    size_t size;
    std::vector<int> ncells;
    std::vector<int> cells;
    std::vector<int> localPadding;
    std::vector<int> localParticleCount;
    std::vector<T> xmin, xmax, ymin, ymax, zmin, zmax;
};

template <typename T>
size_t getNumberOfNodesRec(const Octree<T> &o)
{
    size_t count = 1;
    for (int i = 0; i < o.ncells; i++)
        if(o.cells[i] != nullptr)
            count += getNumberOfNodesRec(*o.cells[i]);
        else
            count++;
    return count;
}

template <typename T>
size_t createLinearOctreeRec(const Octree<T> &o, LinearOctree<T> &l, const int start, const int count, size_t it = 0)
{
    l.localPadding[it] = start;
    l.localParticleCount[it] = count;
    l.ncells[it] = 0;
    l.xmin[it] = o.bbox.xmin;
    l.xmax[it] = o.bbox.xmax;
    l.ymin[it] = o.bbox.ymin;
    l.ymax[it] = o.bbox.ymax;
    l.zmin[it] = o.bbox.zmin;
    l.zmax[it] = o.bbox.zmax;

    size_t padding = 1;

    for (int i = 0; i < 8; i++)
    {
        l.cells[it * 8 + i] = it + padding;
        if(o.cells[i] != nullptr)
        {
            // at least one cell exist
            l.ncells[it] = 8;
            padding += createLinearOctreeRec(*o.cells[i], l, o.start[i], o.count[i], it+padding);
        }
        else
        {
            l.localPadding[it+padding] = o.start[i];
            l.localParticleCount[it+padding] = o.count[i];
            l.ncells[it+padding] = 0;
            padding++;
        }
    }

    return padding;
}

template <typename T>
void createLinearOctree(const Octree<T> &o, LinearOctree<T> &l)
{
    size_t nodeCount = getNumberOfNodesRec(o);

    l.size = nodeCount;
    l.ncells.resize(nodeCount);
    l.cells.resize(8 * nodeCount);
    l.localPadding.resize(nodeCount);
    l.localParticleCount.resize(nodeCount);
    l.xmin.resize(nodeCount);
    l.xmax.resize(nodeCount);
    l.ymin.resize(nodeCount);
    l.ymax.resize(nodeCount);
    l.zmin.resize(nodeCount);
    l.zmax.resize(nodeCount);

    l.localParticleCount[0] = o.x->size();
    l.localPadding[0] = 0;

    createLinearOctreeRec(o, l, 0, o.x->size());
}

} // namespace sphexa
