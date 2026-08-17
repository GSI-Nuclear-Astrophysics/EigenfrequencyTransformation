import numpy as np


def analyze_group_periodicity(
    groups,
    cluster_positions
):

    results = {}


    for gid, g in enumerate(groups):


        clusters = list(
            g["clusters"]
        )


        if len(clusters) < 2:
            continue



        # --------------------------------------------------
        # Frequenzen holen
        # --------------------------------------------------

        frequencies = np.array(
            [
                cluster_positions[c]
                for c in clusters
                if c in cluster_positions
            ]
        )


        clusters = [
            c for c in clusters
            if c in cluster_positions
        ]


        if len(frequencies) < 2:
            continue



        # --------------------------------------------------
        # f_rev aus vorheriger Gruppe
        # --------------------------------------------------

        f_rev_guess = (
            g["harmonic_spacing"]
        )


        if not (
            1.8e6 < f_rev_guess < 2.1e6
        ):
            continue



        # --------------------------------------------------
        # Harmoniken neu bestimmen
        # NICHT übernehmen!
        # --------------------------------------------------

        absolute = (
            frequencies / f_rev_guess
        )


        harmonics = (
            absolute
        )



        # --------------------------------------------------
        # nach Harmoniken sortieren
        # --------------------------------------------------

        order = np.argsort(
            harmonics
        )


        harmonics = harmonics[order]

        frequencies = frequencies[order]

        clusters = (
            np.array(clusters)[order]
            .tolist()
        )



        # --------------------------------------------------
        # Linearer Fit
        # f = f0 + h*f_rev
        # --------------------------------------------------

        fit = np.polyfit(
            harmonics,
            frequencies,
            1
        )


        recovered_frev = fit[0]


        reconstructed = np.polyval(
            fit,
            harmonics
        )


        residuals = (
            frequencies -
            reconstructed
        )


        fit_error = np.mean(
            np.abs(residuals)
        )


        ss_res = np.sum(
            residuals**2
        )


        ss_tot = np.sum(
            (
                frequencies -
                np.mean(frequencies)
            )**2
        )


        r2 = (
            1 -
            ss_res /
            (ss_tot + 1e-12)
        )



        # --------------------------------------------------
        # Ausgabe
        # --------------------------------------------------

        print("\n" + "="*60)
        print(f"Group {gid+1}")
        print("="*60)


        print(
            "Clusters:"
        )

        print(
            clusters
        )


        print(
            "\nPositions"
        )


        for f,h in zip(
            frequencies,
            harmonics
        ):

            print(
                f"{f/1e6:.6f} MHz "
                f" -> harmonic {h}"
            )



        print(
            "\nSpacing"
        )


        spacings = np.diff(
            frequencies
        )


        for d in spacings:

            print(
                f"{d/1e6:.6f} MHz"
            )



        print(
            "\nStatistics"
        )


        print(
            f"Mean spacing : "
            f"{np.mean(spacings)/1e6:.6f} MHz"
        )


        print(
            f"Std spacing  : "
            f"{np.std(spacings)/1e3:.3f} kHz"
        )


        print(
            f"Recovered f_rev : "
            f"{recovered_frev/1e6:.6f} MHz"
        )


        print(
            f"Fit error : "
            f"{fit_error/1e3:.3f} kHz"
        )


        print(
            f"R² : {r2:.8f}"
        )


        print(
            "\nStored:"
        )


        print(
            f"Stored f_rev : "
            f"{g['harmonic_spacing']/1e6:.6f} MHz"
        )



        # --------------------------------------------------
        # speichern
        # --------------------------------------------------

        results[gid] = {

            "clusters":
                clusters,

            "frequencies":
                frequencies,

            "harmonics":
                harmonics,

            "f_rev":
                recovered_frev,

            "fit_error":
                fit_error,

            "r2":
                r2
        }


    return results

import numpy as np
import pandas as pd
# get peaks per cluster and sort them (position wise) 
# only using peaks who can be found in every cluster of the group
# corresponding peaks through the position (first peak to first peak etc)
# revolution frequency for harmonic and revolution frequency = fundamental frequency
def build_peak_families_robust(
    groups,
    raw_peak_results
):

    rows = []

    family_id = 0


    for gid, group in enumerate(groups):

        cluster_ids = group["clusters"]

        # --------------------------------------------------
        # f_rev aus der bereits bestimmten Gruppe
        # --------------------------------------------------

        fundamental = group["harmonic_spacing"]

        if not (
            1.8e6 < fundamental < 2.1e6
        ):
            continue


        cluster_peaks = []


        # --------------------------------------------------
        # Peaks pro Cluster sammeln
        # --------------------------------------------------

        for cid in cluster_ids:

            peaks = [
                r
                for r in raw_peak_results
                if r["cluster_id"] == cid
            ]

            peaks = sorted(
                peaks,
                key=lambda x: x["peak_position"]
            )

            if len(peaks):
                cluster_peaks.append(peaks)


        if len(cluster_peaks) < 2:
            continue


        # --------------------------------------------------
        # Nur Peak-Positionen verwenden, die in allen
        # Clustern vorhanden sind
        # --------------------------------------------------

        min_len = min(
            len(p)
            for p in cluster_peaks
        )


        # --------------------------------------------------
        # Peak-Familien
        # --------------------------------------------------

        for peak_index in range(min_len):

            members = [
                peaks[peak_index]
                for peaks in cluster_peaks
            ]


            freqs = np.array([
                m["peak_position"]
                for m in members
            ])


            # --------------------------------------------------
            # Harmoniken relativ zur gemeinsamen f_rev
            # --------------------------------------------------

            harmonics = freqs / fundamental


            for m, h in zip(
                members,
                harmonics
            ):

                f_peak = m["peak_position"]


                rows.append({

                    "family_id": family_id,

                    "group_id": gid,

                    "cluster_id": m["cluster_id"],

                    "peak_index": peak_index,

                    "f_peak": f_peak,

                    "harmonic": h,

                    "fundamental": fundamental,

                    "folded_frequency":
                        f_peak / h
                        if h != 0
                        else np.nan,

                    "window": m["window"]

                })


            family_id += 1


    return pd.DataFrame(rows)

# since first every peak of every cluster gets mapped back to the first harmonic, 
# the peaks at the fundamental frequency need to be averaged out??? 

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.signal import find_peaks


def merge_close_peaks(peaks, values, min_distance=5, valley_factor=0.7):

    if len(peaks) == 0:
        return []

    peaks = np.array(sorted(peaks))
    merged = []
    i = 0

    while i < len(peaks):

        group = [peaks[i]]
        j = i + 1

        while j < len(peaks):

            if peaks[j] - peaks[i] > min_distance:
                break

            segment = values[peaks[i]:peaks[j] + 1]
            valley = np.min(segment)

            min_height = min(values[peaks[i]], values[peaks[j]])

            if valley > valley_factor * min_height:
                group.append(peaks[j])
                j += 1
            else:
                break

        best = max(group, key=lambda p: values[p])
        merged.append(best)

        i = j

    return np.array(sorted(set(merged)))


def plot_mean_peak_families_styled(
    peak_families,
    frequency,
    spectrum,
    n_points=400,
    prominence_factor=0.02,
    distance=5,
    valley_factor=0.7
):

    fig = go.Figure()

    colors = ["red","blue","green","orange","purple","cyan","magenta","gold"]
    shown = set()

    baseline = 0.005 * np.max(spectrum)

    all_rows = []

    # ============================================================
    # FAMILY LOOP
    # ============================================================
    for fam_id, fam_df in peak_families.groupby("family_id"):

        curves = []

        # -------------------------
        # fold curves
        # -------------------------
        for _, row in fam_df.iterrows():

            left, right = row["window"]
            h = row["harmonic"]

            if h <= 0 or np.isnan(h):
                continue

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

            curves.append((x / h, y))

        if len(curves) == 0:
            continue

        xmin = max(np.min(x) for x,_ in curves)
        xmax = min(np.max(x) for x,_ in curves)

        if xmax <= xmin:
            continue

        x_common = np.linspace(xmin, xmax, n_points)

        Y = []
        for x_folded, y in curves:
            Y.append(np.interp(x_common, x_folded, y))

        y_mean = np.maximum(np.nanmean(Y, axis=0), 0)

        # ============================================================
        # PEAK DETECTION
        # ============================================================
        raw_peaks, _ = find_peaks(
            y_mean,
            prominence=prominence_factor * np.max(y_mean),
            distance=distance
        )

        peaks = merge_close_peaks(
            raw_peaks,
            y_mean,
            min_distance=distance,
            valley_factor=valley_factor
        )

        if len(peaks) == 0:
            peaks = [np.nanargmax(y_mean)]

        # ============================================================
        # TABLE
        # ============================================================
        for p in peaks:
            all_rows.append({
                "family_id": fam_id,
                "peak_position": x_common[p],
                "peak_height": y_mean[p]
            })

        # ============================================================
        # PLOT
        # ============================================================
        color = colors[fam_id % len(colors)]

        x_plot = np.concatenate([[x_common[0]], x_common, [x_common[-1]]])
        y_plot = np.concatenate([[0], np.maximum(y_mean, 0), [0]])

        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"Family {fam_id}" if fam_id not in shown else None,
            showlegend=fam_id not in shown
        ))

        shown.add(fam_id)

        fig.add_trace(go.Scatter(
            x=x_common[peaks],
            y=y_mean[peaks],
            mode="markers",
            marker=dict(size=9, color=color),
            showlegend=False
        ))

    peak_table = pd.DataFrame(all_rows)

    print("\n===== PEAK TABLE =====")
    print(peak_table)

    fig.update_layout(
        title="Mean Peak Families",
        xaxis_title="Frequency (folded)",
        yaxis_title="Amplitude",
        template="plotly_white"
    )

    fig.show()

    return peak_table

