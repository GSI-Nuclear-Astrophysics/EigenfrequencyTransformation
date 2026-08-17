import numpy as np
from scipy.signal import find_peaks


def find_peaks_adaptive(
    frequency,
    amplitude,
    prominence_resonance=0.003,
    distance_resonance=10,
    threshold_background=0.001,
    merge_background_points=7,
    activity_window=300,
    activity_fraction=0.40
):
    # rough estimation of peaks with find_peaks

    rough_peaks, props = find_peaks(
        amplitude,
        prominence=max(
            prominence_resonance * 0.3,
            threshold_background
        ),
        distance=max(
            2,
            distance_resonance // 2
        )
    )


    activity = np.zeros(len(amplitude))

    half = activity_window // 2


    for peak, prom in zip(
        rough_peaks,
        props["prominences"]
    ):

        left = max(
            0,
            peak-half
        )

        right = min(
            len(amplitude),
            peak+half
        )

        activity[left:right] += 1
        activity[left:right] += prom



    kernel = np.ones(activity_window) / activity_window

    activity = np.convolve(
        activity,
        kernel,
        mode="same"
    )


    # resonance area

    center = np.argmax(activity)

    limit = activity_fraction * activity[center]


    left = center
    while left > 0 and activity[left] > limit:
        left -= 1


    right = center
    while right < len(activity)-1 and activity[right] > limit:
        right += 1


    resonance_mask = np.zeros(
        len(amplitude),
        dtype=bool
    )

    resonance_mask[left:right+1] = True



    # peaks in the resonance area 

    resonance_indices = np.where(
        resonance_mask
    )[0]


    peaks_res_local, _ = find_peaks(
        amplitude[resonance_mask],
        prominence=prominence_resonance,
        distance=distance_resonance
    )


    peaks_res = resonance_indices[
        peaks_res_local
    ]



    # peaks in the "background" (not resonance area)

    background_indices = np.where(
        ~resonance_mask
    )[0]


    peaks_background = []


    for i in background_indices:

        if i == 0 or i == len(amplitude)-1:
            continue


        if (
            amplitude[i-1]
            <
            amplitude[i]
            >
            amplitude[i+1]
        ):

            if amplitude[i] > threshold_background:

                peaks_background.append(i)



    peaks_background = np.asarray(
        peaks_background,
        dtype=int
    )



    # merging 

    def merge_peaks_fast(
        peaks,
        amplitude,
        merge_points
    ):

        if len(peaks) == 0:
            return peaks


        peaks = np.sort(peaks)

        merged = []

        group = [
            peaks[0]
        ]


        for p in peaks[1:]:

            if p - group[-1] <= merge_points:

                group.append(p)

            else:

                group = np.asarray(group)

                merged.append(
                    group[
                        np.argmax(
                            amplitude[group]
                        )
                    ]
                )

                group = [p]


        group = np.asarray(group)

        merged.append(
            group[
                np.argmax(
                    amplitude[group]
                )
            ]
        )


        return np.asarray(
            merged,
            dtype=int
        )



    peaks_background = merge_peaks_fast(
        peaks_background,
        amplitude,
        merge_background_points
    )



    # combined Peaks

    peaks_all = np.unique(
        np.concatenate(
            [
                peaks_res,
                peaks_background
            ]
        )
    )


    return (
        peaks_all,
        resonance_mask,
        activity
    )



def get_adaptive_peaks(
    frequency,
    amplitude
):

    peaks, resonance_mask, activity = find_peaks_adaptive(
        frequency,
        amplitude,
        prominence_resonance=0.003,
        distance_resonance=10,
        threshold_background=0.001,
        merge_background_points=7,
        activity_window=300,
        activity_fraction=0.40
    )


    return (
        np.asarray(peaks, dtype=int),
        resonance_mask,
        activity
    )

import plotly.graph_objects as go

def plot_spectrum_with_peaks_plotly(
    frequency,
    amplitude,
    peaks,
    f_min=None,
    f_max=None
):

    fig = go.Figure()

    # Spectrum
    fig.add_trace(go.Scatter(
        x=frequency,
        y=amplitude,
        mode="lines",
        name="Spectrum"
    ))

    fig.add_trace(go.Scatter(
        x=frequency[peaks],
        y=amplitude[peaks],
        mode="markers",
        marker=dict(color="red", size=6),
        name="Peaks"
    ))

    if f_min is not None and f_max is not None:
        fig.update_xaxes(range=[f_min, f_max])

    fig.update_layout(
        title="Spectrum with Peaks",
        xaxis_title="Frequency",
        yaxis_title="Amplitude"
    )

    fig.show()