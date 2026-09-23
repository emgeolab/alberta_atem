# Analysis of `chifact` and Data Uncertainty in SimPEG ATEM Inversion

This document provides a comprehensive technical guide and theoretical insight into how **`chifact`** (target misfit factor) and **data uncertainty** ($\sigma$, standard deviation / floor / relative error) operate, interact, and govern the final resistivity models in SimPEG TDEM inversions.

The analysis is specifically tailored to the workflow, settings, and inversion test results examined in [`notebooks/few_lines_test.ipynb`](file:///Volumes/X31/01.Projects/atem/notebooks/few_lines_test.ipynb) and [`tools/run_inversion_sharp.py`](file:///Volumes/X31/01.Projects/atem/tools/run_inversion_sharp.py).

---

## 1. Mathematical Foundations in SimPEG

### 1.1 Objective Function & Tikhonov Regularization

SimPEG solves the deterministic inverse problem by minimizing an objective function consisting of a data misfit term $\phi_d(m)$ and a regularization (model objective) term $\phi_m(m)$, balanced by the trade-off parameter $\beta$:

$$\phi(m) = \phi_d(m) + \beta \phi_m(m)$$

### 1.2 Data Misfit & Uncertainty Weighting

The weighted least-squares data misfit is defined as:

$$\phi_d(m) = \frac{1}{2} \sum_{i=1}^{N} \left( \frac{d_i^{\text{pred}}(m) - d_i^{\text{obs}}}{\sigma_i} \right)^2 = \frac{1}{2} \| \mathbf{W}_d (\mathbf{d}^{\text{pred}}(m) - \mathbf{d}^{\text{obs}}) \|_2^2$$

where:
* $N$ is the total number of active data observations across all channels and soundings.
* $\sigma_i$ is the standard deviation (uncertainty) assigned to observation $i$.
* $\mathbf{W}_d = \text{diag}(1/\sigma_1, 1/\sigma_2, \dots, 1/\sigma_N)$ is the data weighting matrix.

### 1.3 Statistical Expectation and `chifact`

Under the classical assumption that the measurement errors are independent, zero-mean Gaussian random variables with true standard deviations equal to $\sigma_i$:
* The normalized squared residual $\chi^2 = 2 \phi_d(m)$ follows a chi-squared distribution with $N$ degrees of freedom.
* The mathematical expected value of $\phi_d$ at the true model is:

$$\mathbb{E}[ \phi_d ] \approx \frac{N}{2}$$

SimPEG defines the **target data misfit** using `chifact`:

$$\phi_d^{\text{target}} = \text{chifact} \times \frac{N}{2}$$

* When $\text{chifact} = 1.0$, the algorithm stops cooling $\beta$ when the data misfit reaches the expected noise level ($\phi_d = N/2$, or reduced chi-squared $\chi^2/N = 1.0$).
* When $\text{chifact} > 1.0$, the inversion terminates at a looser data fit, terminating earlier in the $\beta$-cooling schedule.
* When $\text{chifact} < 1.0$, the inversion attempts to fit the data closer than the estimated noise level.

---

## 2. Mathematical Duality: Uncertainty vs. `chifact`

A fundamental property of least-squares inversion is the scaling relationship between $\sigma$ and $\phi_d$:

If all uncertainties $\sigma_i$ are uniformly scaled by a constant factor $c$ ($\sigma_i \to c \cdot \sigma_i$):
$$\phi_d^{\text{new}} = \frac{1}{c^2} \phi_d^{\text{old}}$$

Consequently:
$$\text{Targeting } \text{chifact} = c^2 \text{ with uncertainty } \sigma \iff \text{Targeting } \text{chifact} = 1.0 \text{ with uncertainty } c \cdot \sigma$$

### Example:
* Setting $\text{chifact} = 4.0$ with an assumed $5\%$ relative noise floor is mathematically identical to running with $\text{chifact} = 1.0$ and a $10\%$ relative noise floor ($c = 2$, $c^2 = 4$).
* In [`notebooks/few_lines_test.ipynb`](file:///Volumes/X31/01.Projects/atem/notebooks/few_lines_test.ipynb), the comparisons between:
  - `sharp_2.5chi` ($\sqrt{2.5} \approx 1.58\times$ noise scaling)
  - `sharp_3chi` ($\sqrt{3.0} \approx 1.73\times$ noise scaling)
  - `sharp_4chi` ($\sqrt{4.0} = 2.0\times$ noise scaling)
  demonstrate how adjusting `chifact` acts as an effective scalar multiplier on your assumed noise budget.

---

## 3. Structural Differences: `uncertainty` vs. `chifact`

While a uniform scalar uncertainty change mirrors a `chifact` change, **uncertainty and `chifact` serve distinct roles**:

```
+-----------------------------------------------------------------------------+
|                           DATA UNCERTAINTY (sigma_i)                        |
|   - Relative weight between different channels (early vs. late time)        |
|   - Relative weight between different soundings (clean vs. noisy data)      |
|   - Hard filtering via infinite uncertainty (sigma = np.inf)                |
|   -> Shapes the sensitivity distribution and depth weighting                |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                              CHIFACT (Scalar)                               |
|   - Global stopping threshold: phi_d_target = chifact * (N / 2)             |
|   - Controls how far beta is cooled                                         |
|   - Governs the trade-off between model structure (phi_m) and data fit      |
|   -> Determines overall sharpness/roughness vs. smoothness                  |
+-----------------------------------------------------------------------------+
```

### 3.1 Role of Uncertainty Components in ATEM

In [`notebooks/few_lines_test.ipynb`](file:///Volumes/X31/01.Projects/atem/notebooks/few_lines_test.ipynb#L305-L331), data uncertainty is constructed as:

```python
# 1. Relative error threshold & cutoff
criteria_rerr = 0.05
criteria_uncertainty = 0.05

# 2. Additive floor (calibrated to peak transmitter current & turns)
floors_c = 0.05
floors = 5 * 1e-9 / (df_data_binned["TranPeak"] * n_turns) * floors_c

# 3. Base uncertainty: max(binned std, 5% of |dobs|) + floor
dobs_std = abs(dobs) * criteria_uncertainty
dobs_std[std > dobs_std] = std[std > dobs_std]
dobs_std[cut_off.flatten()] = np.inf  # discard high-noise channels
dobs_std += np.repeat(floors.values, end_channel - start_channel)
```

1. **Relative Error (`criteria_uncertainty = 0.05`)**:
   - Dominates at **early to mid times** where the transient response $dB/dt$ is large ($10^{-7}$ to $10^{-9}\,\text{V/A}\cdot\text{m}^4$).
   - If relative uncertainty is set too low, early channels disproportionately dominate the inversion, overfitting the near-surface and ignoring deeper conductivity.
2. **Noise Floor (`floors`)**:
   - Dominates at **late times** where signal amplitude decays toward the ambient electromagnetic noise level ($10^{-11}$ to $10^{-12}\,\text{V/A}\cdot\text{m}^4$).
   - Without an adequate floor, relative error ($\sigma = 0.05 \cdot |d|$) approaches zero at late times, causing the weights ($W_d = 1/\sigma$) to artificially blow up and generate severe conductive or resistive oscillations at maximum investigation depth.
3. **Cut-off Mask (`np.inf`)**:
   - Bins with high standard deviation (`data_rerr > 0.05`) are set to $\sigma = \infty$, yielding $W_d = 0$. These points exert zero gradient on the model update.

---

## 4. Specific Impact on IRLS (Sharp / L0-Norm) Inversion

In [`tools/run_inversion_sharp.py`](file:///Volumes/X31/01.Projects/atem/tools/run_inversion_sharp.py#L128-L135), the inversion uses:

```python
reg = LaterallyConstrained(
    mesh_reg, 
    mapping=simpeg.maps.IdentityMap(nP=nP),
    alpha_s=0,
    alpha_r=alpha_r,
    alpha_z=alpha_z,
    active_edges=indActiveEdges,
    norms=np.array([2., 0., 0.])  # L2 in smallness, L0 in horizontal, L0 in vertical
)

update_irls = simpeg.directives.Update_IRLS(
    coolingFactor=2,
    coolingRate=1,
    f_min_change=1e-5,
    max_irls_iterations=40,
    chifact_start=1.2,
    chifact_target=1.2,    
)
```

The Iteratively Reweighted Least Squares (IRLS) workflow operates in two phases:
1. **Smooth Phase (Standard L2)**: Beta is cooled until $\phi_d \le \text{chifact\_start} \times (N/2)$.
2. **IRLS Reweighting Phase (L0/L1 Sharp)**: The regularization scales edge weights based on model gradients to focus contrasts into discrete, sharp boundaries.

### Interaction with `chifact`:
* **If `chifact_start` is set too low (e.g. 1.0 when data has 10% unmodeled noise)**:
  - The optimizer cannot reach `chifact_start` during the L2 phase.
  - Beta cools to tiny values trying to fit non-1D geological noise or instrument drift.
  - The inversion never enters IRLS, or enters it with severe oscillatory artifacts in the model. IRLS then "sharpens" these noise oscillations into sharp, artificial layer contacts.
* **If `chifact` is set moderately higher (e.g. 2.5 - 3.0, as tested in `sharp_2.5chi` and `sharp_3chi`)**:
  - The smooth L2 inversion stops cooling $\beta$ as soon as the broad structural features are fit.
  - IRLS successfully kicks in, converting the smooth gradients of real geological packages (e.g. Quaternary sediments, sand channels, and bedrock) into clean, sharp interfaces without sharpening noise spikes.
* **If `chifact` is set too high (e.g. $\ge 4.0$)**:
  - The inversion stops prematurely. Subtle resistive units (e.g., thin coal seams or interbedded silts) remain underfit and appear muted or smeared.

---

## 5. Summary of Observed Behaviors in `few_lines_test.ipynb`

In Cell 24 of [`notebooks/few_lines_test.ipynb`](file:///Volumes/X31/01.Projects/atem/notebooks/few_lines_test.ipynb#L550-L620), the normalized misfit evolution shows:

| Run Configuration | Initial L2 Fit ($\phi_d / N$) | Post-IRLS Final Fit ($\phi_d / N$) | Model Characteristics |
| :--- | :--- | :--- | :--- |
| **`sharp_2.5chi`** | $\approx 0.95$ (iter 4) | $\approx 1.42$ - $1.66$ | Crisp layer boundaries; fits late-time decay well; slight risk of sounding-to-sounding lateral roughness if noise is elevated. |
| **`sharp_3.0chi`** | $\approx 0.98$ (iter 4) | $\approx 1.65$ - $1.87$ | **Optimal balance**: high lateral continuity across flight lines; stable layer interfaces; resilient against flight-height variation. |
| **`sharp_4.0chi`** | $\approx 1.25$ (iter 3) | $\approx 2.13$ - $2.88$ | Conservative model; very smooth laterally; minor underfitting of subtle decay curve variations. |

---

## 6. Practical Workflow Guidelines for ATEM Inversion

1. **Keep Uncertainty Geophysically Realistic**:
   - Base relative error ($5\% - 8\%$) on repeat flight lines or binning standard deviations.
   - Base the noise floor ($5 \times 10^{-9} / (\text{TranPeak} \times \text{turns}) \times c_{\text{floor}}$) on the standard deviation of raw off-time noise channels when the transmitter is off or at late times where signal has decayed.
   - Do not use uncertainty solely as a tuning knob for model roughness; use uncertainty to balance the *relative* weights across time gates.

2. **Use `chifact` to Control Model Complexity**:
   - Keep `chifact_start` and `chifact_target` in `Update_IRLS` between **1.5 and 3.0** for field ATEM data.
   - Field airborne EM data rarely satisfies pure 1D assumptions due to 3D geometry, bird swing, and lateral conductivity gradients. An empirical `chifact` of $2.0 \sim 3.0$ prevents the 1D forward solver from forcing 3D effects into vertical layer stripes.

3. **Check Quality via Diagnostic Plots**:
   - **Normalized Residuals**: Check that $(d^{\text{pred}} - d^{\text{obs}})/\sigma$ is centered at 0 with no systematic bias across time gates.
   - **Sounding 1D Profiles**: Ensure layer resistivities do not hit upper/lower bound limits ($1/\rho \to \text{extreme}$).
   - **Line Sections (XZ)**: Verify that layer contacts track known geological markers (e.g., bedrock elevation, well lithology) without artificial vertical striping along the flight line.
