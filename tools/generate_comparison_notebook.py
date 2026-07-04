import os
import json

notebook_path = "/Volumes/X31/01-Projects/atem/notebooks/well_inversion_comparison.ipynb"

# Define notebook cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Comparison of Well Log Resistivity and 1D TEM Inversion Results\n",
            "\n",
            "This notebook performs spatial matching between water wells/coal holes and binned TEM flight line soundings, extracts resistivity curves from LAS files, aligns them in depth, and plots comparative curves of well log resistivity vs the 1D inversion model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import geopandas as gpd\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import dill\n",
            "import lasio as ls\n",
            "import matplotlib.pyplot as plt\n",
            "from scipy.spatial import cKDTree\n",
            "\n",
            "# Change working directory to project root if running from notebooks directory\n",
            "if os.path.basename(os.getcwd()) == \"notebooks\":\n",
            "    os.chdir(\"..\")\n",
            "print(f\"Current Working Directory: {os.getcwd()}\")\n",
            "\n",
            "# Add tools path to import binning function\n",
            "sys.path.append(\"tools\")\n",
            "from tools import binning\n",
            "\n",
            "from simpeg.electromagnetics.utils.em1d_utils import get_vertical_discretization\n",
            "from simpeg.utils import plot_1d_layer_model"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "WELL_DATA_DIR = \"data/well_data\"\n",
            "WATER_WELL_SHP = os.path.join(WELL_DATA_DIR, \"WaterWell_HQ_LAS/WaterWell_HQ_LAS/Shapefile/WWLAS_CLC_HQ.shp\")\n",
            "COAL_HOLE_SHP = os.path.join(WELL_DATA_DIR, \"CoalHoles_HQ_LAS/CoalHoles_HQ_LAS/ShapeFile/CoalHole_CLC_HQ.shp\")\n",
            "\n",
            "# Load shapefiles\n",
            "ww_gdf = gpd.read_file(WATER_WELL_SHP)\n",
            "ch_gdf = gpd.read_file(COAL_HOLE_SHP)\n",
            "\n",
            "# Convert to UTM Zone 12N (EPSG:26912) to match flight lines\n",
            "ww_utm12 = ww_gdf.to_crs(epsg=26912)\n",
            "ch_utm12 = ch_gdf.to_crs(epsg=26912)\n",
            "\n",
            "print(f\"Loaded {len(ww_gdf)} Water Wells and {len(ch_gdf)} Coal Holes.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def find_matching_sounding(well_name, well_gdf_utm, df_binned, pik_path, dheader):\n",
            "    \"\"\"\n",
            "    Finds the nearest sounding in the binned dataframe spatially,\n",
            "    then runs a local search in the pik file to match the exact sounding index.\n",
            "    \"\"\"\n",
            "    if 'Name' in well_gdf_utm.columns:\n",
            "        well_row = well_gdf_utm[well_gdf_utm['Name'] == well_name].iloc[0]\n",
            "    else:\n",
            "        well_row = well_gdf_utm[well_gdf_utm['LAS_FI_NAM'] == well_name].iloc[0]\n",
            "        \n",
            "    well_xy = np.array([well_row.geometry.x, well_row.geometry.y])\n",
            "    \n",
            "    # 1. Spatial lookup in df_binned\n",
            "    tree = cKDTree(df_binned[['x_wgs84', 'y_wgs84']].values)\n",
            "    dist, idx_binned = tree.query(well_xy)\n",
            "    \n",
            "    # 2. Local search in .pik file\n",
            "    with open(pik_path, \"rb\") as f:\n",
            "        out_dict = dill.load(f)\n",
            "    last_iter = max(out_dict.keys())\n",
            "    m = out_dict[last_iter]['m']\n",
            "    dpred = out_dict[last_iter]['dpred']\n",
            "    n_layers = 22\n",
            "    n_soundings_pik = int(m.shape[0] / n_layers)\n",
            "    \n",
            "    dpred_reshaped = np.abs(dpred.reshape((n_soundings_pik, n_layers))) + 1e-15\n",
            "    dobs_abs = np.abs(df_binned[dheader].values[idx_binned]) + 1e-15\n",
            "    \n",
            "    search_start = max(0, idx_binned - 500)\n",
            "    search_end = min(n_soundings_pik, idx_binned + 500)\n",
            "    \n",
            "    dpred_search = dpred_reshaped[search_start:search_end]\n",
            "    diffs = np.sum((np.log10(dpred_search) - np.log10(dobs_abs))**2, axis=1)\n",
            "    best_match_local_idx = np.argmin(diffs)\n",
            "    idx_pik = search_start + best_match_local_idx\n",
            "    \n",
            "    rho_sounding = 1. / np.exp(m.reshape((n_soundings_pik, n_layers))[idx_pik])\n",
            "    line = df_binned.iloc[idx_binned]['Line']\n",
            "    dtm = df_binned.iloc[idx_binned]['dtm']\n",
            "    \n",
            "    print(f\"Matched Well {well_name} -> Binned Sounding Index: {idx_binned}, Pik Index: {idx_pik}\")\n",
            "    print(f\"Distance: {dist:.1f} m | Line: {line} | Topography Elevation: {dtm:.1f} m\")\n",
            "    \n",
            "    return idx_pik, rho_sounding, dist, line, dtm"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def load_well_las(well_name, well_type):\n",
            "    \"\"\"\n",
            "    Loads LAS file and extracts valid depth and resistivity/conductivity logs.\n",
            "    Converts conductivity to resistivity and re-scales depth if needed.\n",
            "    \"\"\"\n",
            "    if well_type == \"WaterWell\":\n",
            "        path = f\"data/well_data/WaterWell_HQ_LAS/WaterWell_HQ_LAS/LAS_Files/{well_name}.las\"\n",
            "    else:\n",
            "        path = f\"data/well_data/CoalHoles_HQ_LAS/CoalHoles_HQ_LAS/LAS_Files/Litholog_{well_name}.las\"\n",
            "        if not os.path.exists(path):\n",
            "            path = f\"data/well_data/CoalHoles_HQ_LAS/CoalHoles_HQ_LAS/LAS_Files/Litholog_{well_name}_A.las\"\n",
            "            \n",
            "    las = ls.read(path)\n",
            "    df_las = las.df()\n",
            "    \n",
            "    depth = df_las.index.values\n",
            "    # Detect units (assume meters by default, convert feet to meters if index units are feet)\n",
            "    units = las.index_unit\n",
            "    if units is None and len(las.curves) > 0:\n",
            "        units = las.curves[0].unit\n",
            "    if units is not None:\n",
            "        units = str(units).upper()\n",
            "        if 'FT' in units or 'FEET' in units:\n",
            "            depth = depth * 0.3048\n",
            "            print(f\"  Converted depth from feet to meters for well {well_name}\")\n",
            "        \n",
            "    res = None\n",
            "    # Look for resistivity/conductivity curves\n",
            "    for col in ['RES', 'RES1', 'RES2', 'SFR', 'CILD']:\n",
            "        if col in df_las.columns:\n",
            "            res_val = df_las[col].values\n",
            "            if col == 'CILD':\n",
            "                # Convert conductivity (mS/m) to resistivity (ohm-m): R = 1000 / C\n",
            "                res_val = 1000.0 / (res_val + 1e-6)\n",
                "                print(f\"  Converted CILD (conductivity) to resistivity for {well_name}\")\n",
            "            res = res_val\n",
            "            break\n",
            "            \n",
            "    if res is None:\n",
            "        raise ValueError(f\"No resistivity or conductivity curve found in well {well_name}\")\n",
            "        \n",
            "    # Mask out negative or invalid values\n",
            "    mask = (depth >= 0) & (res > 0) & (res < 10000)\n",
            "    return depth[mask], res[mask]"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def plot_comparison(well_name, well_type, area, line, dist, depth_well, res_well, res_sounding, dtm):\n",
            "    \"\"\"\n",
            "    Plots the comparative analysis: well log resistivity and 1D TEM inversion model vs depth.\n",
            "    \"\"\"\n",
            "    hz = get_vertical_discretization(21, 2, 1.17)\n",
            "    \n",
            "    fig, ax = plt.subplots(figsize=(6, 8))\n",
            "    \n",
            "    # Plot well log\n",
            "    ax.plot(res_well, depth_well, color='black', label=f'Well Log ({well_name})', linewidth=1.5)\n",
            "    \n",
            "    # Plot Simpeg 1D inversion model\n",
            "    plot_1d_layer_model(hz, res_sounding, ax=ax, color='crimson', label='1D Inversion Model', linewidth=2)\n",
            "    \n",
            "    # Configure axes\n",
            "    ax.set_xscale('log')\n",
            "    ax.invert_yaxis()\n",
            "    ax.set_xlabel(r'Resistivity ($\\Omega \\cdot m$)', fontsize=12)\n",
            "    ax.set_ylabel('Depth (m)', fontsize=12)\n",
            "    ax.set_title(f'Resistivity vs Depth Comparison\\nArea: {area} | Well: {well_name} (Dist: {dist:.1f}m) | Line: {line}', fontsize=12)\n",
            "    ax.grid(True, which='both', linestyle='--', alpha=0.5)\n",
            "    ax.legend(loc='lower left', fontsize=10)\n",
            "    \n",
            "    # Set y-limits to the depth of the well log + 20m buffer\n",
            "    max_depth = min(depth_well.max(), 320.0)\n",
            "    ax.set_ylim([max_depth + 10.0, 0])\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. NE Area Comparison: Water Well `111913-a` vs NE Inversion\n",
            "\n",
            "We load the NE area survey data, bin it using the exact parameters, and locate the nearest flight sounding to Water Well `111913-a` (distance: **12.5m**)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "area = \"NE\"\n",
            "pik_path = \"data/archives/NE_inv_results_atem_full.pik\"\n",
            "\n",
            "# Run exact binning for NE\n",
            "print(\"Running exact binning for NE...\")\n",
            "res_ne = binning(dx=50.0, area=area)\n",
            "df_binned_ne = pd.DataFrame(\n",
            "    data=np.vstack(res_ne[\"values\"]), columns=[\"Line\", \"distance\"] + res_ne[\"picker\"][1:]\n",
            ")\n",
            "for col in df_binned_ne.columns:\n",
            "    if col != \"Line\":\n",
            "        df_binned_ne[col] = pd.to_numeric(df_binned_ne[col])\n",
            "\n",
            "# Find matching sounding and plot\n",
            "idx_pik, res_sounding, dist, line, dtm = find_matching_sounding(\n",
            "    \"111913-a\", ww_utm12, df_binned_ne, pik_path, res_ne[\"dheader\"]\n",
            ")\n",
            "depth_well, res_well = load_well_las(\"111913-a\", \"WaterWell\")\n",
            "\n",
            "plot_comparison(\"111913-a\", \"WaterWell\", area, line, dist, depth_well, res_well, res_sounding, dtm)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. SE Area Comparison 1: Water Well `153845-a` vs SE Inversion\n",
            "\n",
            "We load the SE area survey data and locate the nearest flight sounding to Water Well `153845-a` (distance: **4.0m**)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "area = \"SE\"\n",
            "pik_path = \"data/archives/SE_inv_results_atem_full.pik\"\n",
            "\n",
            "# Run exact binning for SE\n",
            "print(\"Running exact binning for SE...\")\n",
            "res_se = binning(dx=50.0, area=area)\n",
            "df_binned_se = pd.DataFrame(\n",
            "    data=np.vstack(res_se[\"values\"]), columns=[\"Line\", \"distance\"] + res_se[\"picker\"][1:]\n",
            ")\n",
            "for col in df_binned_se.columns:\n",
            "    if col != \"Line\":\n",
            "        df_binned_se[col] = pd.to_numeric(df_binned_se[col])\n",
            "\n",
            "# Find matching sounding and plot\n",
            "idx_pik, res_sounding, dist, line, dtm = find_matching_sounding(\n",
            "    \"153845-a\", ww_utm12, df_binned_se, pik_path, res_se[\"dheader\"]\n",
            ")\n",
            "depth_well, res_well = load_well_las(\"153845-a\", \"WaterWell\")\n",
            "\n",
            "plot_comparison(\"153845-a\", \"WaterWell\", area, line, dist, depth_well, res_well, res_sounding, dtm)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. SE Area Comparison 2: Coal Hole `TH24-83` vs SE Inversion\n",
            "\n",
            "We compare Coal Hole `TH24-83` with the SE inversion results (distance: **51.0m**)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Find matching sounding and plot for Coal Hole TH24-83 using the already binned SE data\n",
            "idx_pik, res_sounding, dist, line, dtm = find_matching_sounding(\n",
            "    \"TH24-83\", ch_utm12, df_binned_se, pik_path, res_se[\"dheader\"]\n",
            ")\n",
            "depth_well, res_well = load_well_las(\"TH24-83\", \"CoalHole\")\n",
            "\n",
            "plot_comparison(\"TH24-83\", \"CoalHole\", area, line, dist, depth_well, res_well, res_sounding, dtm)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. NW Area Comparison: Water Well `111800-a` vs NW Inversion\n",
            "\n",
            "We load the NW area survey data and locate the nearest flight sounding to Water Well `111800-a` (distance: **180.5m**)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "area = \"NW\"\n",
            "pik_path = \"data/archives/NW_inv_results_atem_full.pik\"\n",
            "\n",
            "# Run exact binning for NW\n",
            "print(\"Running exact binning for NW...\")\n",
            "res_nw = binning(dx=50.0, area=area)\n",
            "df_binned_nw = pd.DataFrame(\n",
            "    data=np.vstack(res_nw[\"values\"]), columns=[\"Line\", \"distance\"] + res_nw[\"picker\"][1:]\n",
            ")\n",
            "for col in df_binned_nw.columns:\n",
            "    if col != \"Line\":\n",
            "        df_binned_nw[col] = pd.to_numeric(df_binned_nw[col])\n",
            "\n",
            "# Find matching sounding and plot\n",
            "idx_pik, res_sounding, dist, line, dtm = find_matching_sounding(\n",
            "    \"111800-a\", ww_utm12, df_binned_nw, pik_path, res_nw[\"dheader\"]\n",
            ")\n",
            "depth_well, res_well = load_well_las(\"111800-a\", \"WaterWell\")\n",
            "\n",
            "plot_comparison(\"111800-a\", \"WaterWell\", area, line, dist, depth_well, res_well, res_sounding, dtm)"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "atem",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Ensure directories exist
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated comparison notebook at: {notebook_path}")
