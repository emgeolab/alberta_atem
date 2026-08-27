import dill
import scipy
import numpy as np
import pandas as pd
from rich import print
from types import SimpleNamespace
import argparse
import json

# SimPEG Modules
import simpeg
from simpeg.electromagnetics.utils.em1d_utils import get_vertical_discretization
from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

try:
    from pymatsolver import PardisoSolver as Solver
except NameError:
    from pymatsolver import Solver

    print("PardisoSolver is not available. Falling back to Solver.")

from simpeg.electromagnetics.utils.em1d_utils import set_mesh_1d
from discretize import SimplexMesh
from simpeg.regularization.laterally_constrained import LaterallyConstrained

import tools


def main(
    miter: int,
    areas: list,
    output_prefix: str,
    ncpu: int,
    alpha_r: float,
    alpha_z: float,
):
    print(f"ncpu:{ncpu}")
    print(f"alpha_r:{alpha_r}")
    print(f"alpha_z:{alpha_z}")
    # Load worst files from all areas and merge them
    # worst_line = []
    # for area in areas:
    #     worst_path = f"./data/{area}_worst_50_smoothed_rmse.json"
    #     try:
    #         with open(worst_path, "r") as f:
    #             area_worst = [i for i in json.load(f)[0].values()]
    #             worst_line.extend(area_worst)
    #         print(f"Loaded {len(area_worst)} worst lines from {worst_path}")
    #     except Exception as e:
    #         print(f"Warning: could not load worst line file '{worst_path}': {e}")

    # Deduplicate worst_line
    # worst_line = list(set(worst_line))
    # print(f"Total unique worst lines merged: {len(worst_line)}")

    # Decide the channels to be used for inversion
    istart_channel: int = 3
    iend_channel: int = 25

    # Data binning
    dx = 50.0

    print("01. Data Binning\n")

    all_values = []
    all_values_std = []
    all_soundings = []

    times = None
    n_turns = None
    dheader = None
    picker = None

    for area in areas:
        print(f"Binning area: {area}...")
        # tmp: dict = tools.binning_normal(dx=dx, area=area)
        tmp: dict = tools.binning(dx=dx, area=area)

        if times is None:
            times = tmp["times"]
            n_turns = tmp["n_turns"]
            dheader = tmp["dheader"]
            picker = tmp["picker"]

        all_values.extend(tmp["values"])
        all_values_std.extend(tmp["values_std"])
        all_soundings.extend(tmp["soundings"])

    df_data_binned = pd.DataFrame(
        data=np.vstack(all_values), columns=["Line", "distance"] + picker[1:]
    )
    df_data_std_binned = pd.DataFrame(
        data=np.vstack(all_values_std), columns=["bheight"] + dheader
    )

    for col in df_data_binned.columns:
        if col != "Line":
            df_data_binned[col] = pd.to_numeric(df_data_binned[col])

    print(f"\tBinned soundings: {len(df_data_binned):,}\n")

    # Criteria for bad data (uncertainty correction)
    data_ = df_data_binned[dheader].values.astype(float)
    data_rerr = (
        df_data_std_binned[dheader].values / np.abs(df_data_binned[dheader].values)
    ).astype(float)
    # bad_line_id = df_data_binned["Line"].isin(worst_line).values
    # bad_line_mask = np.repeat(bad_line_id[:, None], data_.shape[1], axis=1)

    criteria_rerr: float = 0.05  # Select binned data having high std.
    criteria_uncertainty: float = (
        0.05  # Select how much uncertainty will be used for normalization.
    )
    # bad_line_uncertainty: float = 0.30

    channel_id = np.tile(np.arange(data_.shape[1]), (data_.shape[0], 1))
    cut_off = (data_rerr > criteria_rerr) * (channel_id >= 0)
    floors_c: float = 0.05
    floors = 5 * 1e-9 / (df_data_binned["TranPeak"] * n_turns) * floors_c

    del data_, data_rerr
    dobs = df_data_binned[dheader].values.flatten()  # Binned Data
    std = df_data_std_binned[
        dheader
    ].values.flatten()  # Standard deviation of Binned Data
    dobs_std = abs(dobs) * criteria_uncertainty  # 0.05 * Binned Data
    dobs_std[std > dobs_std] = std[
        std > dobs_std
    ]  # Standard deviation having smaller than 0.05 * dobs will be replaced with 0.05 * dobs.
    # dobs_std[bad_line_mask.flatten()] = (
    #     abs(dobs[bad_line_mask.flatten()]) * bad_line_uncertainty
    # )
    dobs_std[cut_off.flatten()] = (
        np.inf
    )  # Filter out bad data by setting their standard deviation to infinity
    dobs_std += np.repeat(floors.values, iend_channel - istart_channel)  # Add floors

    print("02. Discretization\n")
    # Set topography and transmitter heights
    topography = df_data_binned[["x_wgs84", "y_wgs84", "dtm"]].values
    source_heights = df_data_binned["bheight"].values
    thickness = get_vertical_discretization(21, 2, 1.17)

    # Waveform
    start_time = -1.74e-3
    peak_time = -0.84e-3
    off_time = 0.0

    waveform = tdem.sources.TriangularWaveform(start_time, off_time, peak_time)

    current_times = np.linspace(start_time, off_time)
    currents = [waveform.eval(t) for t in current_times]

    radius: float = 5.0

    input_data_dict = {
        "topography": topography.astype(float),
        "source_heights": source_heights.astype(float),
        "thickness": thickness,
        "time_input_currents": current_times,
        "input_currents": currents,
        "times": times,
        "data": dobs,
        "data_std": dobs_std,
    }
    inp = SimpleNamespace(**input_data_dict)

    source_locations = np.c_[
        inp.topography[:, 0],
        inp.topography[:, 1],
        inp.topography[:, 2] + inp.source_heights,
    ]
    receiver_locations = np.c_[
        inp.topography[:, 0],
        inp.topography[:, 1],
        inp.topography[:, 2] + inp.source_heights,
    ]
    n_sounding = source_locations.shape[0]

    source_list = []
    receiver_orientation = "z"
    source_orientation = "z"

    print("\n03. Set survey\n")
    # Survey
    for i_sounding in range(n_sounding):
        # waveform = tdem.sources.PiecewiseLinearWaveform(inp.time_input_currents, inp.input_currents)
        source_location = source_locations[i_sounding, :]
        receiver_location = receiver_locations[i_sounding, :]

        # Receiver list

        dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location,
            inp.times,
            receiver_orientation,
        )

        # Make a list containing all receivers even if just one

        # Must define the transmitter properties and associated receivers

        source_list.append(
            tdem.sources.CircularLoop(
                [dbzdt_receiver],
                location=source_location,
                waveform=waveform,
                radius=radius,
                i_sounding=i_sounding,
                orientation=source_orientation,
            )
        )

    survey = tdem.Survey(source_list)

    # Layer
    hz = np.r_[inp.thickness, inp.thickness[-1]]

    n_layer = len(hz)
    nP = n_sounding * n_layer
    sigma_map = maps.ExpMap(nP=nP)

    # Simulation
    simulation = tdem.Simulation1DLayeredStitched(
        survey=survey,
        thicknesses=inp.thickness,
        sigmaMap=sigma_map,
        topo=inp.topography,
        parallel=True,
        n_cpu=ncpu,
        verbose=False,
        solver=Solver,
    )

    # Create data object
    data_object = simpeg.data.Data(survey, dobs=dobs, standard_deviation=inp.data_std)
    dmis = simpeg.data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

    print("04. Start Inversion\n")

    # nData
    inds_active_dobs = dobs.shape[0] - cut_off.sum()
    print(
        f"  Percentage of the active data = {inds_active_dobs:,}/{len(dobs):,}={inds_active_dobs.sum() / len(dobs) * 100:,.0f}%"
    )

    tri = scipy.spatial.Delaunay(inp.topography[:, :2])
    mesh_radial = SimplexMesh(tri.points, tri.simplices)
    mesh_vertical = set_mesh_1d(hz)
    mesh_reg = [mesh_radial, mesh_vertical]

    # INFO: `indActiveEdges` decides the maximum distance to apply regularization along the horizontal direction. Since the line spacing is 750 m, if we apply 500 m for maximum distance, the regularization in direction to tie line wouldn't be applied.
    inds, indActiveEdges = get_active_edge_indices_with_distance(
        mesh_radial, mesh_vertical, maximum_distance=1500.0
    )  # TODO: Try to adjust maximum_distance.

    # Sharp inversion: zero-nrom (Counts of non-zero misfits)
    reg = LaterallyConstrained(
        mesh_reg,
        mapping=simpeg.maps.IdentityMap(nP=nP),
        alpha_s=0.0,
        alpha_r=alpha_r,
        alpha_z=alpha_z,
        active_edges=indActiveEdges,
        norms=np.array([2.0, 0.0, 0.0]),
    )
    reg.gradient_type = "components"

    opt = simpeg.optimization.ProjectedGNCG(maxIter=miter, maxIterCG=50)
    invProb = simpeg.inverse_problem.BaseInvProblem(dmis, reg, opt)
    beta = simpeg.directives.BetaSchedule(
        coolingFactor=2, coolingRate=1
    )  # TODO: Adjust cooling rate
    betaest = simpeg.directives.BetaEstimate_ByEig(beta0_ratio=1)  # TODO: Adjust beta0
    target = simpeg.directives.TargetMisfit(chifact=1)
    precond = simpeg.directives.UpdatePreconditioner()
    save_model_dict = simpeg.directives.SaveOutputDictEveryIteration()
    save_model_dict.outDict = {}

    update_irls = simpeg.directives.Update_IRLS(
        coolingFactor=2,
        coolingRate=1,
        f_min_change=1e-5,
        max_irls_iterations=40,
        chifact_start=1.2,
        chifact_target=1.2,
    )
    update_irls.coolEpsFact = 2.0

    inv = simpeg.inversion.BaseInversion(
        invProb,
        directiveList=[
            update_irls,
            betaest,
            beta,
            precond,
            # target,
            save_model_dict,
        ],
    )
    invProb.counter = opt.counter = simpeg.utils.Counter()
    opt.LSshorten = 0.5
    opt.remember("xc")
    res0 = 10.0
    m0 = np.ones(nP) * np.log(1.0 / res0)
    print(f"{res0=}")

    # Run inversion
    inv.run(m0)

    print("05. Save Results\n")
    # Save results
    name: str = f"./outputs/{output_prefix}_inv_results_atem_full.pik"
    dill.dump(save_model_dict.outDict, open(name, "wb"))

    with open(f"./outputs/{output_prefix}_soundings.json", "w") as f:
        json.dump(all_soundings, f)


def get_active_edge_indices_with_distance(
    mesh_radial, mesh_vertical, maximum_distance=1000
):
    nz = mesh_vertical.n_cells
    edge_lengths = mesh_radial.edge_lengths
    inds = edge_lengths < maximum_distance
    indActiveEdges = np.tile(inds.reshape([-1, 1]), nz).flatten()
    return inds, indActiveEdges


if __name__ == "__main__":
    # 1. Generate parser object
    parser = argparse.ArgumentParser(description="Multi-Area Merged ATEM Inversion")
    # 2. Add arguments
    parser.add_argument("-i", "--iter", type=int, default=40, help="Max Iteration")
    parser.add_argument(
        "-a",
        "--areas",
        type=str,
        default="NW,NE,SE,SW,tielines",
        help="Comma-separated survey areas to merge and invert together",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="merged",
        help="Prefix to use for saved output files",
    )
    parser.add_argument("-n", "--ncpu", type=int, help="No. of cpu", required=True)
    parser.add_argument(
        "-r", "--alphar", type=float, help="Horizontal constraint", required=True
    )
    parser.add_argument(
        "-z", "--alphaz", type=float, help="Vertical constraint", required=True
    )
    # 3. Parse arguments
    args = parser.parse_args()

    # Process comma-separated areas list
    area_list = [a.strip() for a in args.areas.split(",")]

    # 4. Execution
    print(f"{args.iter=}\n{area_list=}\n{args.output=}\n{args.ncpu=}\n")
    main(
        miter=args.iter,
        areas=area_list,
        output_prefix=args.output,
        ncpu=args.ncpu,
        alpha_r=args.alphar,
        alpha_z=args.alphaz,
    )
