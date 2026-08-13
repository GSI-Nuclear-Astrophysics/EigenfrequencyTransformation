import numpy as np
import plotly.graph_objects as go


def build_continuous_spectrum_from_mean_families(
    peak_families,
    frequency,
    spectrum,
    n_points=60000,
    baseline_factor=0.005,
    baseline_floor=1e-12
):

    print("=" * 60)
    print("INPUT DEBUG")
    print("=" * 60)
    print("Spectrum max :", np.max(spectrum))
    print("Spectrum min :", np.min(spectrum))
    print("Spectrum id  :", id(spectrum))

    baseline = baseline_factor * np.max(spectrum)

    print("Baseline     :", baseline)

    family_curves = []

    # ============================================================
    # 1. FAMILY CURVES
    # ============================================================
    for fam_id, fam_df in peak_families.groupby("family_id"):

        curves = []

        print("\n-----------------------------------------")
        print(f"Family {fam_id}")

        for _, row in fam_df.iterrows():

            left, right = row["window"]
            h = row["harmonic"]

            if h <= 0 or np.isnan(h):
                continue

            # baseline extension
            l = left
            while l > 0 and spectrum[l] > baseline:
                l -= 1
            if l > 0:
                l -= 1

            r_idx = right
            while r_idx < len(spectrum)-1 and spectrum[r_idx] > baseline:
                r_idx += 1
            if r_idx < len(spectrum)-1:
                r_idx += 1

            x = frequency[l:r_idx]
            y = np.maximum(spectrum[l:r_idx], 0)

            print(
                f"h={h:6.2f} | "
                f"len={len(y):3d} | "
                f"max={np.max(y):.4f} | "
                f"sum={np.sum(y):.4f}"
            )

            curves.append((x / h, y))

        if len(curves) == 0:
            continue

        all_xmins = np.array([np.min(x) for x, _ in curves])
        all_xmaxs = np.array([np.max(x) for x, _ in curves])

        xmin_raw = np.percentile(all_xmins, 5)
        xmax_raw = np.percentile(all_xmaxs, 95)

        pad = 0.2 * (xmax_raw - xmin_raw)

        xmin = xmin_raw - pad
        xmax = xmax_raw + pad

        if xmax <= xmin:
            continue

        x_common = np.linspace(xmin, xmax, n_points)

        Y = []

        for x_folded, y in curves:
            Y.append(np.interp(x_common, x_folded, y))

        y_mean = np.nanmean(Y, axis=0)
        y_mean = np.maximum(y_mean, baseline_floor)

        print(
            f"Family mean: max={np.max(y_mean):.4f}, "
            f"sum={np.sum(y_mean):.4f}"
        )

        family_curves.append((x_common, y_mean))

    if len(family_curves) == 0:
        return None

    x_min = min(c[0][0] for c in family_curves)
    x_max = max(c[0][-1] for c in family_curves)

    x_global = np.linspace(x_min, x_max, n_points)

    stack = []

    for x_c, y_c in family_curves:

        y_interp = np.interp(
            x_global,
            x_c,
            y_c,
            left=baseline_floor,
            right=baseline_floor
        )

        stack.append(y_interp)

    stack = np.array(stack)

    print("\n====================================================")
    print("STACK")
    print("====================================================")

    for i, y in enumerate(stack):
        print(
            f"Family {i}: "
            f"max={np.max(y):.4f} "
            f"sum={np.sum(y):.4f}"
        )

    spectrum_cont = np.nanmean(stack, axis=0) * len(stack)

    print("\nContinuous before normalization")
    print("max =", np.max(spectrum_cont))
    print("sum =", np.sum(spectrum_cont))

    spectrum_cont /= np.max(spectrum_cont)

    print("\nContinuous after normalization")
    print("max =", np.max(spectrum_cont))
    print("sum =", np.sum(spectrum_cont))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_global,
        y=spectrum_cont,
        mode="lines",
        line=dict(color="black", width=2),
        name="Continuous spectrum"
    ))

    fig.update_layout(
        title="Continuous Spectrum",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        template="plotly_white"
    )

    fig.show()

    return x_global, spectrum_cont

import numpy as np
import pandas as pd
from IPython.display import display
# analysis with fwhm etc. and pairwise frequency differences 
def build_frequency_analysis_table(peak_families, frequency, spectrum):

    rows = []

    for _, entry in peak_families.iterrows():

        gid = entry["group_id"]
        cid = entry["cluster_id"]

        harmonic = entry["harmonic"]

        if harmonic <= 0 or np.isnan(harmonic):
            continue

        left, right = entry["window"]

        x = frequency[left:right] / harmonic
        y = spectrum[left:right]

        mask = ~np.isnan(y)
        if np.sum(mask) < 3:
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        peak_idx = np.nanargmax(y_valid)
        peak_pos_folded = x_valid[peak_idx]

        peak_height = y_valid[peak_idx]
        half_max = peak_height / 2

        left_idx = peak_idx
        while left_idx > 0 and y_valid[left_idx] > half_max:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(y_valid) - 1 and y_valid[right_idx] > half_max:
            right_idx += 1

        if right_idx <= left_idx:
            continue

        fwhm_folded = x_valid[right_idx] - x_valid[left_idx]
        sigma_folded = fwhm_folded / (2 * np.sqrt(2 * np.log(2)))

        peak_pos_physical = peak_pos_folded * harmonic

        rows.append({
            "group_id": gid,
            "cluster_id": cid,

            "harmonic": harmonic,

            # folded domain
            "peak_position_folded": peak_pos_folded,
            "fwhm_folded": fwhm_folded,
            "sigma_folded": sigma_folded,

            # physical domain
            "peak_position_Hz": peak_pos_physical,
            "peak_position_MHz": peak_pos_physical / 1e6,

            "fwhm_Hz": fwhm_folded * harmonic,
            "sigma_Hz": sigma_folded * harmonic
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df, pd.DataFrame()

    df = df.sort_values("peak_position_Hz").reset_index(drop=True)

    #  PAIRWISE DIFFERENCES

    diff_rows = []

    positions = df["peak_position_Hz"].values
    sigmas = df["sigma_Hz"].values

    gids = df["group_id"].values
    cids = df["cluster_id"].values
    harmonics = df["harmonic"].values

    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            delta_f = positions[j] - positions[i]
            delta_sigma = np.sqrt(sigmas[i]**2 + sigmas[j]**2)

            rel_df = delta_f / positions[i] if positions[i] != 0 else np.nan

            diff_rows.append({
                "group_1": gids[i],
                "group_2": gids[j],

                "cluster_1": cids[i],
                "cluster_2": cids[j],

                "harmonic_1": harmonics[i],
                "harmonic_2": harmonics[j],

                "f1_Hz": positions[i],
                "f2_Hz": positions[j],

                "delta_f_Hz": delta_f,
                "abs_delta_f_Hz": abs(delta_f),

                "sigma_delta_f_Hz": delta_sigma,
                "relative_df_over_f": rel_df
            })

    df_diff = pd.DataFrame(diff_rows)

    df_diff = df_diff.sort_values("abs_delta_f_Hz").reset_index(drop=True)

    # OUTPUT

    print("\n==============================")
    print("PEAK POSITIONS (CONSISTENT)")
    print("==============================")

    display(df[[
        "group_id",
        "cluster_id",
        "harmonic",
        "peak_position_MHz",
        "sigma_Hz"
    ]])

    print("\n==============================")
    print("PAIRWISE DIFFERENCES")
    print("==============================")

    display(df_diff)

    return df, df_diff
