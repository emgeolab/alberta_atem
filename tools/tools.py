import json
import numpy as np
import pandas as pd


def binning(
    dx: float,
    area: str,
    istart_line: int = 0,
    iend_line: int = None,
    istart_channel: int = 3,
    iend_channel: int = 25,
):

    # Load the configuration file
    conf = json.load(open("./data/atem.json"))
    n_turns = conf["n_turns"]
    times = np.asarray(conf["channels"])[istart_channel:iend_channel] * 1e-6

    path: str = f"./data/11-024_Alberta_{area}.csv"
    dheader: list = [f"zoff30[{i}]" for i in range(istart_channel, iend_channel)]
    picker: list = [
        "Line",
        "bheight",
        "TranPeak",
        "x_wgs84",
        "y_wgs84",
        "flight",
        "pwrline",
        "dtm",
    ] + dheader

    # Load data
    raws = pd.read_csv(path)[picker]

    # Unit conversion and normalization
    normalizer = (-1e-9) / (raws["TranPeak"].values * n_turns).reshape(-1, 1)
    raws[[f"zoff30[{i}]" for i in range(istart_channel, iend_channel)]] = (
        raws[[f"zoff30[{i}]" for i in range(istart_channel, iend_channel)]] * normalizer
    )

    # Extract Line number
    line_no = list(raws["Line"].unique())
    istart: int = istart_line
    iend: int = iend_line

    # Remove none data
    negative_mask = raws["bheight"] < 0
    raws = raws.copy()[~negative_mask]
    raws.dropna(subset=["x_wgs84", "y_wgs84"], inplace=True)
    raws.fillna(1e-20, inplace=True)

    # Data binning 세팅
    values = []
    values_std = []
    soundings = []

    # Filtering parameters
    sigma = dx / 2.0
    window_radius = dx * 1.5  # Distance 1.1%

    for i_line, line in enumerate(line_no[istart:iend]):
        df_line = raws[raws["Line"] == line].copy()
        if len(df_line) == 0:
            continue

        # Calculate distance along the "Line"
        xy = df_line[["x_wgs84", "y_wgs84"]].to_numpy()
        distance = np.sqrt(((xy - xy[0, :]) ** 2).sum(axis=1))
        df_line.insert(0, "distance", distance)

        max_distance = distance.max()

        # Binning distances
        # Determine the no. of soundings per bin.
        if max_distance % dx == 0:
            n_sounding = int(max_distance / dx)
        else:
            n_sounding = int(np.floor(max_distance / dx) + 1)

        # Create bins and assign each sounding to a bin
        bins = np.arange(n_sounding) * dx
        n_sounding = len(bins)
        soundings.append(n_sounding)

        line_values = []
        line_std = []
        line_size = []

        # Select the target distance for each bin
        for tar_bin in bins:
            # Filter raw data within the window radius
            mask = (df_line["distance"] >= tar_bin - window_radius) & (
                df_line["distance"] <= tar_bin + window_radius
            )
            sub_df = df_line[mask]

            # If no data is found within the window radius, select the closest data point
            if len(sub_df) == 0:
                closest_idx = (df_line["distance"] - tar_bin).abs().idxmin()
                sub_df = df_line.loc[[closest_idx]]

            # Gaussian weighting (not simple average)
            dist_diff = sub_df["distance"].to_numpy() - tar_bin
            weights = np.exp(-(dist_diff**2) / (2 * sigma**2))
            weights /= weights.sum()

            # Extract numeric columns for weighted sum.
            numeric_cols = ["distance"] + picker[1:]
            features = sub_df[numeric_cols].to_numpy().astype(float)

            # Calculate weighted sum
            weighted_mean = (features * weights.reshape(-1, 1)).sum(axis=0)

            # Calculate weighted standard deviation
            std_cols = ["bheight"] + dheader
            std_features = sub_df[std_cols].to_numpy().astype(float)

            if len(sub_df) > 1:
                mean_std_features = (std_features * weights.reshape(-1, 1)).sum(axis=0)
                weighted_var = (
                    (std_features - mean_std_features) ** 2 * weights.reshape(-1, 1)
                ).sum(axis=0)
                weighted_std = np.sqrt(weighted_var)
            else:
                weighted_std = np.zeros(std_features.shape[1])

            binned_row = np.zeros(len(numeric_cols) + 1, dtype=object)
            binned_row[0] = line
            binned_row[1:] = weighted_mean

            line_values.append(binned_row)
            line_std.append(weighted_std)
            line_size.append(len(sub_df))

        values.append(np.array(line_values, dtype=object))
        values_std.append(np.array(line_std))

    del raws

    return {
        "times": times,
        "soundings": soundings,
        "n_turns": n_turns,
        "values": values,
        "values_std": values_std,
        "dheader": dheader,
        "picker": picker,
    }


if __name__ == "__main__":
    print("Hi")
