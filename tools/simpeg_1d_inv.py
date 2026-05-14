import simpeg.electromagnetics.time_domain as tdem
from simpeg.utils import plot_1d_layer_model, download, mkvc
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
)
from discretize import TensorMesh
import numpy as np
import json

def main(channels, dobs, dalt, floors):
    # Define the survey
    # Source
    source_location = np.array([0., 0., dalt])
    source_orientation:str = "z"
    source_current:float = 1.
    source_radius:float = 5.

    # Receiver
    receiver_location = np.array([0. , 0., dalt])
    receiver_orientation:str = "z"

    receiver_list = []
    receiver_list.append(
        tdem.receivers.PointMagneticFluxTimeDerivative(receiver_location, times, receiver_orientation)
    )

    # Waveform
    start_time = -1.74e-3
    peak_time = -0.84e-3
    off_time = 0.0
    waveform = tdem.sources.TriangularWaveform(
    start_time=start_time, peak_time=peak_time, off_time=off_time
    )

    # Source
    source_list = [
        tdem.sources.CircularLoop(
            receiver_list=receiver_list,
            location=source_location,
            orientation=source_orientation,
            current=source_current,
            radius=source_radius,
            waveform=waveform,
        )
    ]

    # Survey
    survey = tdem.Survey(source_list=source_list)

    # Uncertainties
    uncertainties = 0.05 * np.abs(dobs) + floors

    # Data object
    data_object = data.Data(survey=survey, dobs=dobs, standard_deviation=uncertainties)
    

if __name__ == "__main__":
    conf = json.load(open("../data/atem.json"))
    channels = np.array(conf["channels"]) * 1e-6
    n_turns = conf['n_turns']

    # Load data
    area:str = "NE"
    path:str = f"../data/11-024_Alberta_{data_area}.csv"
    dheader:list = [f"zoff30[{i}]" for i in range(30)]
    picker:list = ["Line", "bheight", "TranPeak", "x_wgs84", "y_wgs84", "flight", 'pwrline'] + dheader
    dobs = pd.read_csv

    main()
