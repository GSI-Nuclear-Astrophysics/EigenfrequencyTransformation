import numpy as np
def normalize_peak_shape(x, y, mode="l2"):
    # no negative y- Values
    # area of the peak -> creating a probability form
    # l2 = Energy normalization -> comparing peaks independent from the strength of the signal
    # making peaks more comparable
    x = np.asarray(x)
    y = np.maximum(np.asarray(y), 0)

    if len(y) < 3:
        return y

    if mode == "l1":
        area = np.trapezoid(y, x)
        if area > 0:
            y = y / area

    elif mode == "l2":
        norm = np.linalg.norm(y)
        if norm > 0:
            y = y / norm

    return y  

def build_shape_spectrum_from_clusters(
    results,
    frequency,
    f_min=None,
    f_max=None,
    mode="l2"
):

    spectrum = np.zeros_like(frequency)

    for r in results:

        x = r["x"]
        y = r["y"]

        y = np.maximum(y, 0)
        y = normalize_peak_shape(x, y, mode=mode)

        left = np.searchsorted(frequency, x[0])
        right = left + len(y)

        if right <= len(spectrum):
            spectrum[left:right] += y

    return spectrum

import numpy as np


def build_raw_spectrum_from_clusters(
    results,
    frequency,
    f_min=None,
    f_max=None
):

    spectrum = np.zeros_like(frequency)

    for r in results:

        x = r["x"]

        # intensity
        y = r["y_raw"]

        y = np.maximum(y, 0)

        left = np.searchsorted(
            frequency,
            x[0]
        )

        right = left + len(y)

        if right <= len(spectrum):
            spectrum[left:right] += y

    return spectrum

