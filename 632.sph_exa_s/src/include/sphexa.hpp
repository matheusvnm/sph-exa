#pragma once

#include "sph/kernels.hpp"
#include "sph/findNeighbors.hpp"
#include "sph/density.hpp"
#include "sph/equationOfState.hpp"
#include "sph/momentumAndEnergyIAD.hpp"
#include "sph/timestep.hpp"
#include "sph/positions.hpp"
#include "sph/totalEnergy.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef SPEC_OPENACC
#include "openacc.h"
#endif
#if defined(SPEC_OPENMP) || defined(SPEC_OPENMP_TARGET)
#include "omp.h"
#endif

#include "DistributedDomain.hpp"
#include "Domain.hpp"
#include "Octree.hpp"
#include "BBox.hpp"

#include "ArgParser.hpp"
#include "config.hpp"
#include "timer.hpp"
