#include <iostream>
#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "SqPatch.hpp"

#ifdef SPEC
#define TIME(name)
#else
#define TIME(name)                                                                                                                         \
    if (d.rank == 0) timer.step(name)
#endif

using namespace std;
using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);
    int cubeSide = parser.getInt("-n", 50);
    int maxStep = parser.getInt("-s", 10);
    int writeFrequency = parser.getInt("-w", -1);
    int blockSize = 1;
#ifdef SPEC_OMPSS
        blockSize = BLOCK_SIZE;
        if (!((blockSize != 0) && ((blockSize & (blockSize - 1)) == 0)) || blockSize > cubeSide * cubeSide * cubeSide)
        {   
            printf("Error! The block size must be a power of two number lower than the cube total size.\n");
            return 1;
        }
#endif

#ifdef _JENKINS
    maxStep = 0;
    writeFrequency = -1;
#endif

    typedef double Real;
    typedef Octree<Real> Tree;
    typedef SqPatch<Real> Dataset;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
#endif

    Dataset d(cubeSide, blockSize);
    DistributedDomain<Real> distributedDomain;
    Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);

    vector<int> clist(d.count);
    for (unsigned int i = 0; i < clist.size(); i++)
        clist[i] = i;

    Timer timer;
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    { 
        int nnn = d.count;
        timer.start();
        // Verificar por quê tasks não funciona aqui.
        d.resize(d.count); 
        distributedDomain.distribute(clist, d); // Taskwait
        TIME("domain::build");
        distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m); // Taskwait
        TIME("mpi::synchronizeHalos");
        domain.buildTree(d);
        TIME("BuildTree");
        sph::findNeighbors(domain.tree, clist, d);
        TIME("FindNeighbors");
       

        if(clist.size() > 0) sph::computeDensity<Real>(clist, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(clist, d); }
        TIME("Density");
        sph::computeEquationOfState<Real>(clist, d);
        TIME("EquationOfState");   
        
        // Inserir taskwait aqui!
        distributedDomain.resizeArrays(d.count, &d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c); // Discard halos
        distributedDomain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c); // Need tasking implementation here
        TIME("mpi::synchronizeHalos");

        if(clist.size() > 0) sph::computeIAD<Real>(clist, d);
        TIME("IAD");
        distributedDomain.resizeArrays(d.count, &d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33); // Discard halos
        distributedDomain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33); // Need tasking implementation here
        TIME("mpi::synchronizeHalos");  
        
        if(clist.size() > 0) sph::computeMomentumAndEnergyIAD<Real>(clist, d);
        TIME("MomentumEnergyIAD");
        sph::computeTimestep<Real>(clist, d);
        TIME("Timestep");
        sph::computePositions<Real>(clist, d);
        TIME("UpdateQuantities");
        sph::computeTotalEnergy<Real>(clist, d);
        TIME("EnergyConservation"); 
        

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            std::ofstream dump("dump" + to_string(d.iteration) + ".txt");
            d.writeData(clist, dump);
            TIME("writeFile");
            dump.close();
        }
        //d.writeConstants(d.iteration, totalNeighbors, constants);

        timer.stop();
        if (d.rank == 0) cout << "=== Total time for iteration(" << d.iteration << ") " << timer.duration() << "s" << endl << endl;
    }

    //constants.close();
    if (d.rank == 0)
    {
        std::ofstream final_output("sqpatch.dat");
        final_output << d.etot << endl;
        final_output.close();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
