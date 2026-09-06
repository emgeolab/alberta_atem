import numpy as np
import dill
import warnings
warnings.filterwarnings("ignore")

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem
from pymatsolver import PardisoSolver
import simpeg
from types import SimpleNamespace

import dill
import sys

n_cpu = sys.argv[1]
alpha_s = float(sys.argv[2])
rho_ini = float(sys.argv[3])
rho_ref = float(sys.argv[4])

print (n_cpu, alpha_s, rho_ini, rho_ref)

inversion_type = 'smooth'


input_data_dict = dill.load(open("./input_data_clark_creek.pik", "rb"))
inp = SimpleNamespace(**input_data_dict)
radius = np.sqrt(1/np.pi)
source_locations = np.c_[inp.topography[:,0], inp.topography[:,1], inp.topography[:,2]+inp.source_heights]
receiver_locations = np.c_[inp.topography[:,0], inp.topography[:,1],  inp.topography[:,2]+inp.source_heights]
n_sounding = source_locations.shape[0]

source_list = []
receiver_orientation = 'z'
source_orientation = 'z'
for i_sounding in range(n_sounding):    
    waveform = tdem.sources.PiecewiseLinearWaveform(inp.time_input_currents, inp.input_currents)
    source_location = source_locations[i_sounding, :]
    receiver_location = receiver_locations[i_sounding, :]

    # Receiver list

    dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location, inp.times, "z",
    )

    # Make a list containing all receivers even if just one

    # Must define the transmitter properties and associated receivers

    source_list.append(tdem.sources.CircularLoop(
        [dbzdt_receiver],
        location=source_location,
        waveform=waveform,
        radius=radius,
        i_sounding=i_sounding,
    )
    )

survey = tdem.Survey(source_list)
hz = np.r_[inp.thickness, inp.thickness[-1]]

n_layer = len(hz)
nP = n_sounding * n_layer
sigma_map = maps.ExpMap(nP=nP)

simulation = tdem.Simulation1DLayeredStitched(
    survey=survey, 
    thicknesses=inp.thickness, 
    sigmaMap=sigma_map,
    topo=inp.topography, 
    parallel=True, 
    n_cpu=n_cpu, 
    verbose=False, 
    solver=PardisoSolver,
)

n_time = inp.times.size


# Create data ojbect
data_object = simpeg.data.Data(survey, dobs=inp.data, standard_deviation=inp.data_std)
dmis = simpeg.data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
inds_active_dobs = inp.data_std!=np.inf
print (f"Percentage of the active data = {inds_active_dobs.sum()}/{len(inp.data)}={inds_active_dobs.sum()/len(inp.data)*100:.0f}%")

from simpeg.electromagnetics.utils.em1d_utils import set_mesh_1d
import scipy
from discretize import SimplexMesh
from simpeg.regularization.laterally_constrained import LaterallyConstrained

tri = scipy.spatial.Delaunay(inp.topography[:,:2])
mesh_radial = SimplexMesh(tri.points, tri.simplices)
mesh_vertical = set_mesh_1d(hz)
mesh_reg = [mesh_radial, mesh_vertical]

def get_active_edge_indices_with_distance(mesh_radial, mesh_vertical, maximum_distance=1000):
    nz = mesh_vertical.n_cells
    edge_lengths = mesh_radial.edge_lengths
    inds = edge_lengths < maximum_distance
    indActiveEdges = np.tile(inds.reshape([-1,1]), nz).flatten()
    return inds, indActiveEdges

inds, indActiveEdges = get_active_edge_indices_with_distance(
    mesh_radial, mesh_vertical, maximum_distance=150.
)
reference_model = np.log(1./rho_ref) * np.ones(nP)

reg = LaterallyConstrained(
    mesh_reg, 
    mapping=simpeg.maps.IdentityMap(nP=nP),
    alpha_s = alpha_s,
    alpha_r = 1.,
    alpha_z = 1./2.,
    active_edges=indActiveEdges,
    reference_model=reference_model
)

opt = simpeg.optimization.ProjectedGNCG(maxIter=20, maxIterCG=50)
invProb = simpeg.inverse_problem.BaseInvProblem(dmis, reg, opt)
beta = simpeg.directives.BetaSchedule(coolingFactor=2, coolingRate=1)
betaest = simpeg.directives.BetaEstimate_ByEig(beta0_ratio=1.)
target = simpeg.directives.TargetMisfit(chifact=1)
precond = simpeg.directives.UpdatePreconditioner()
save_model_dict = simpeg.directives.SaveOutputDictEveryIteration()
save_model_dict.outDict = {}
save_model = simpeg.directives.SaveModelEveryIteration()

inv = simpeg.inversion.BaseInversion(
    invProb, 
    directiveList=[
        betaest, 
        beta, 
        precond,
        target,
        save_model_dict
    ]
)
invProb.counter = opt.counter = simpeg.utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
m0 = np.ones(nP) * np.log(1./rho_ini)
mest = inv.run(m0)
dill.dump(save_model_dict.outDict, open(f"./inversion_results_{inversion_type}_{alpha_s:.1e}_{rho_ini:.0f}_{rho_ref:.0f}.pik", "wb"))

