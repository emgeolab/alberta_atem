# Time Channel Smoothing Method

This note summarizes the data smoothing and error scoring workflow used in `notebooks/07.time_channel_smoothing_v1.ipynb`.

## Data Preparation

The ATEM data are loaded from the NE survey file and normalized in the same way as the inversion workflow.

The selected channels are:

```text
zoff30[3] to zoff30[24]
```

Each flight line is binned at 50 m spacing. For each bin, the notebook stores:

- mean data values for each time channel
- standard deviation for each time channel
- x-y location
- flight height
- power-line monitor value
- line ID and distance along line

The main data matrix is:

```python
data_binned = df_data_binned[dheader].values
```

Its shape is:

```text
n_sounding x n_time_channel
```

## Data Reliability Weights

The smoothing uses both spatial proximity and data reliability.

First, a channel-wise relative error is estimated from the binned standard deviation:

```python
data_rerr = data_std_binned / abs(data_binned)
```

Data with relative error larger than the cutoff are treated as unreliable:

```python
relative_error_cutoff = 0.03
bad_data_mask = data_rerr > relative_error_cutoff
```

An uncertainty is assigned using the larger of the bin standard deviation and a minimum relative uncertainty:

```python
minimum_relative_uncertainty = 0.05
base_uncertainty = max(data_std_binned, abs(data_binned) * minimum_relative_uncertainty)
```

A small noise floor is added:

```python
floor_fraction = 0.05
noise_floor = 5e-9 / (TranPeak * n_turns) * floor_fraction
data_uncertainty = base_uncertainty + noise_floor
```

Bad data are not removed from the original data matrix, but their uncertainty is set to infinity:

```python
data_uncertainty[bad_data_mask] = inf
```

The reliability weight is inverse variance:

```python
data_reliability_weight = 1 / data_uncertainty**2
```

So reliable data contribute more strongly to smoothing, while high-error data contribute weakly or not at all.

## KDTree Spatial Smoothing

The notebook smooths each sounding using nearby soundings found by a KDTree.

Current smoothing parameters:

```python
k_nearest_points = 50
idw_epsilon = 1
idw_power_early = 1.5
idw_power_late = 2.0
min_effective_neighbors = 5.0
```

The IDW power varies by time channel:

```python
idw_power_by_time = linspace(idw_power_early, idw_power_late, n_time_channel)
```

This means:

- early time channels use lower IDW power and receive stronger smoothing
- late time channels use higher IDW power and receive weaker smoothing

For each sounding and each neighbor, the spatial weight is:

```python
spatial_weight = 1 / (distance + idw_epsilon) ** idw_power_by_time
```

The final smoothing weight is:

```python
total_weight = spatial_weight * data_reliability_weight
```

The smoothed data are computed as a weighted average:

```python
smoothed_data = sum(total_weight * neighbor_data) / sum(total_weight)
```

The notebook also checks whether the weighted average is controlled by enough meaningful neighbors:

```python
effective_neighbor_count = sum(total_weight)**2 / sum(total_weight**2)
```

If `effective_neighbor_count` is smaller than `min_effective_neighbors`, the original data value is retained instead of using the smoothed value. This reduces unstable smoothing near map boundaries or data gaps where only a few neighbors dominate the result.

The smoothing is evaluated in chunks to reduce memory use:

```python
smoothing_chunk_size = 50_000
```

The chunking only limits memory use. The KDTree is built from the full binned dataset, so each chunk still searches neighbors from the full survey area.

## Line-Level Scores

The notebook compares original data `D` and smoothed data `S`, then aggregates the residuals by flight line.

The channel-wise relative residual is:

```python
relative_residual = (D - S) / (abs(S) + channel_floor)
```

The channel floor is estimated from the smoothed data:

```python
channel_floor = percentile(abs(smoothed_data), 5, axis=0)
```

This avoids unstable values where the smoothed response is close to zero.

For each flight line, the residual RMS values are summarized into line-level metrics.

For each line, the notebook computes:

- number of soundings
- number of valid soundings
- RMS score
- median score
- 90th percentile
- 95th percentile
- maximum score
- fraction above the line residual threshold
- median valid channel count
- line center coordinates
- line distance range

The main line-level metric is:

```python
line_l2_rms = sqrt(mean(sounding_residual_rms**2))
```

This measures the overall line-level strength of the smoothing residual.

The summary table is:

```python
line_l2_summary
```

It is sorted by `line_l2_rms` in descending order.

## Visualization

The notebook includes several visual checks.

### Original Time-Channel Map

This plots the binned data in x-y space for a selected `zoff30` channel.

The selected line is shown with larger points, while other lines are shown as smaller background points.

### Original vs Smoothed Map

This compares original and KDTree-smoothed data side by side for a selected time channel and line.

### Line Profile

This compares original and smoothed data along a selected flight line as a function of distance.

It can show either all channels or one selected `zoff30` channel.

The same plot also overlays normalized `bheight` and `pwrline` values on a twin y-axis. These monitor values are min-max normalized within the selected line, so their trends can be compared with the original-smoothed data difference even though their physical units are different.

### Observed Time-Channel Map And Inversion Depth Slice

The final diagnostic view shows two panels side by side:

- selected observed time-channel map
- selected inversion resistivity depth slice

The high-score lines are selected from `line_l2_rms`.

The observed time-channel map colors the data by the selected `zoff30` channel. The top-N high-score lines are plotted with larger markers on the same color scale. This checks whether the numerically picked lines also show line effects in the observed time-domain data.

The inversion panel shows a resistivity depth slice from the saved inversion result. The top-N high-score lines are overlaid on the resistivity map using the resistivity value at the selected depth.

This checks whether the same high-score lines are connected to line-like artifacts in the inversion product.

The inversion resistivity layers are cached after loading the inversion result, so changing the depth or iteration slider does not repeatedly recompute the full resistivity model.

## Interpretation

A high `line_l2_rms` suggests that a line is systematically different from its locally smoothed expectation.

These lines are candidates for increased uncertainty in a later inversion workflow.

However:

- high `line_l2_rms` suggests line-wide behavior
- high `line_l2_p95` with moderate RMS suggests localized line segments
- high `fraction_above_line_residual_threshold` suggests many anomalous soundings on the same line

These metrics should be checked against maps, line profiles, and inversion depth slices before assigning line-level uncertainty multipliers.
