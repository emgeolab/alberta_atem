import json
import dill
import scipy
import numpy as np
import pandas as pd
from rich import print
from types import SimpleNamespace

# SimPEG Modules
import simpeg
from simpeg.electromagnetics.utils.em1d_utils import get_vertical_discretization
from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem
try:
    from pymatsolver import PardisoSolver as Solver
except:
    from pymatsolver import Solver
    print("PardisoSolver is not available. Falling back to Solver.")

from simpeg.electromagnetics.utils.em1d_utils import set_mesh_1d
from discretize import SimplexMesh
from simpeg.regularization.laterally_constrained import LaterallyConstrained



def main(miter: int):
    
    # Decide the channels to be used for inversion 
    istart_channel:int = 3
    iend_channel:int = 25

    # Load the configuration file
    conf = json.load(open("./data/atem.json"))
    times = np.asarray(conf['channels'])[istart_channel:iend_channel] * 1e-6
    n_turns = conf['n_turns']

    # Decide data path and the header of the data to be used for inversion
    area:str = "NE"
    path:str = f"./data/11-024_Alberta_{area}.csv"
    dheader:list = [f"zoff30[{i}]" for i in range(istart_channel,iend_channel)]
    picker:list = ["Line", "bheight", "TranPeak", "x_wgs84", "y_wgs84", "flight", 'pwrline', 'dtm'] + dheader 

    # Load data
    raws = pd.read_csv(path)[picker]

    # Unit conversion and normalization
    normalizer = (-1e-9)/ (raws["TranPeak"].values * n_turns).reshape(-1, 1)
    raws[[f"zoff30[{i}]" for i in range(istart_channel, iend_channel)]] = raws[[f"zoff30[{i}]" for i in range(istart_channel, iend_channel)]] * normalizer

    # Extract Line number 
    line_no = list(raws["Line"].unique())
    istart:int = 0
    iend:int = None
    # iend:int = 1
    index = raws["Line"] == line_no[istart]
    print(f"{line_no[istart]=}")

    # Reomove none data.
    raws.dropna(subset=["y_wgs84"], inplace=True) # Remove rows with NaN values in y_wgs84
    raws.fillna(1e-20, inplace=True) # Replace remaining NaN values with a small number (1e-20) to avoid issues in calculations

    # Data binning 
    dx = 50.
    values = []
    values_std = []
    soundings = []

    print("==========Data Binning Start==========")
    for i_line, line in enumerate(line_no[istart:iend]):
        df_line = raws[raws['Line']==line]

        # Calculate distance along the "Line"
        xy = df_line[["x_wgs84", "y_wgs84"]].to_numpy()
        distance = np.sqrt(((xy-xy[0,:])**2).sum(axis=1))
        # print(f"Number of NaN values in distance: {np.isnan(distance).sum()}")
        max_distance = distance.max()

        # Determine the no. of soundings per bin.
        if max_distance % dx ==0:
            n_sounding = int(max_distance / dx)
        else:
            n_sounding = int(np.round(max_distance / dx) + 1)

        # Create bins and assign each sounding to a bin
        bins = np.arange(n_sounding) * dx
        df_line.insert(0, 'distance', distance)
        # Bin distances
        df_line['bin'] = pd.cut(df_line['distance'], bins=bins)
        # Compute statistics per bin
        binned = (
            df_line.groupby('bin', observed=False)
                [['distance'] + picker[1:]]
                .mean()
        )
        binned.insert(0, 'Line', line)
        binned_std = (
            df_line.groupby('bin', observed=False)
                [['bheight'] + dheader]
                .std()
        )
        values.append(binned.values)
        values_std.append(binned_std.values)
        soundings.append(n_sounding)

    print("==========Data Binning End==========")

    del raws
    df_data_binned = pd.DataFrame(data=np.vstack(values), columns=['Line', 'distance'] + picker[1:])
    df_data_std_binned = pd.DataFrame(data=np.vstack(values_std), columns=['bheight'] + dheader)

    print(f"{len(soundings)=}")

    # Criteria for bad data (uncertainty correction)
    data_ = df_data_binned[dheader].values.astype(float)
    data_rerr = (df_data_std_binned[dheader].values / np.abs(df_data_binned[dheader].values)).astype(float)

    criteria_rerr:float = 0.03 # Select binned data having highg std.
    criteria_uncertainty:float = 0.05  # Select how much uncertainty will be used for normalization.

    channel_id = np.tile(np.arange(data_.shape[1]), (data_.shape[0], 1))
    cut_off = (data_rerr > criteria_rerr) * (channel_id>=0)
    floors_c:float = 0.05
    floors = 5*1e-9/ (df_data_binned["TranPeak"] * n_turns) * floors_c

    del data_, data_rerr
    dobs = df_data_binned[dheader].values.flatten() # Binned Data
    std = df_data_std_binned[dheader].values.flatten() # Standard deviation of Binned Data
    dobs_std = abs(dobs) * criteria_uncertainty # 0.05 * Binned Data
    dobs_std[std > dobs_std] = std[std > dobs_std] # Standard deviation having smaller than 0.05 * dobs will be replaced with 0.05 * dobs.
    dobs_std[cut_off.flatten()] = np.inf # Filter out bad data by setting their standard deviation to infinity
    dobs_std += np.repeat(floors.values, iend_channel - istart_channel) # Add floors

    # Set topography and transmitter heights
    topography = df_data_binned[['x_wgs84', 'y_wgs84', 'dtm']].values
    source_heights = df_data_binned['bheight'].values
    thickness = get_vertical_discretization(21, 2, 1.17)

    # Waveform
    start_time = -1.74e-3
    peak_time = -0.84e-3
    off_time = 0.0

    waveform =  tdem.sources.TriangularWaveform(start_time, off_time, peak_time)

    current_times = np.linspace(start_time, off_time)
    currents = [waveform.eval(t) for t in current_times]

    radius:float = 5.

    input_data_dict = {
        "topography": topography.astype(float),
        "source_heights": source_heights.astype(float),
        "thickness": thickness,
        "time_input_currents":current_times,
        "input_currents":currents,
        "times":times,    
        "data":dobs,
        "data_std":dobs_std,    
    }
    inp = SimpleNamespace(**input_data_dict)

    source_locations = np.c_[inp.topography[:,0], inp.topography[:,1], inp.topography[:,2] + inp.source_heights]
    receiver_locations = np.c_[inp.topography[:,0], inp.topography[:,1],  inp.topography[:,2] + inp.source_heights]
    n_sounding = source_locations.shape[0]

    source_list = []
    receiver_orientation = 'z'
    source_orientation = 'z'

    for i_sounding in range(n_sounding):    
        # waveform = tdem.sources.PiecewiseLinearWaveform(inp.time_input_currents, inp.input_currents)
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
        n_cpu=10, 
        verbose=False, 
        solver=Solver,
    )

    n_time = inp.times.size

    # Create data ojbect
    data_object = simpeg.data.Data(survey, dobs=dobs, standard_deviation=inp.data_std)
    dmis = simpeg.data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

    # nData
    inds_active_dobs = dobs.shape[0] - cut_off.sum()
    print (f"Percentage of the active data = {inds_active_dobs:,}/{len(dobs):,}={inds_active_dobs.sum()/len(dobs)*100:,.0f}%")

    tri = scipy.spatial.Delaunay(inp.topography[:,:2])
    mesh_radial = SimplexMesh(tri.points, tri.simplices)
    mesh_vertical = set_mesh_1d(hz)
    mesh_reg = [mesh_radial, mesh_vertical]

    inds, indActiveEdges = get_active_edge_indices_with_distance(
        mesh_radial, mesh_vertical, maximum_distance=500.
    )

    reg = LaterallyConstrained(
        mesh_reg, 
        mapping=simpeg.maps.IdentityMap(nP=nP),
        alpha_s = 0.,
        alpha_r = 1.,
        alpha_z = 1./2.,
        active_edges=indActiveEdges
    )

    opt = simpeg.optimization.ProjectedGNCG(maxIter=miter, maxIterCG=50)
    invProb = simpeg.inverse_problem.BaseInvProblem(dmis, reg, opt)
    beta = simpeg.directives.BetaSchedule(coolingFactor=2, coolingRate=1)
    betaest = simpeg.directives.BetaEstimate_ByEig(beta0_ratio=1.)
    target = simpeg.directives.TargetMisfit(chifact=1)
    precond = simpeg.directives.UpdatePreconditioner()
    save_model_dict = simpeg.directives.SaveOutputDictEveryIteration()
    save_model_dict.outDict = {}

    inv = simpeg.inversion.BaseInversion(
        invProb, 
        directiveList=[
            betaest, 
            beta, 
            precond,
            # target, 
            save_model_dict
        ]
    )
    invProb.counter = opt.counter = simpeg.utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')
    m0 = np.ones(nP) * np.log(1./10.)
    mest = inv.run(m0)

    name:str = "./data/inv_results_atem_full.pik"
    dill.dump(save_model_dict.outDict, open(f"{name}", "wb"))

def get_active_edge_indices_with_distance(mesh_radial, mesh_vertical, maximum_distance=1000):
    nz = mesh_vertical.n_cells
    edge_lengths = mesh_radial.edge_lengths
    inds = edge_lengths < maximum_distance
    indActiveEdges = np.tile(inds.reshape([-1,1]), nz).flatten()
    return inds, indActiveEdges


if __name__ == "__main__":
    main(miter=20)
