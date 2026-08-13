import numpy as np
# creating a copy of every cluster 
# every peak has the same length
def build_cluster_template_shape(
    cluster_id,
    raw_peak_results,
    target_len=200
):

    parts = []

    for r in raw_peak_results:

        if r["cluster_id"] != cluster_id:
            continue

        y = np.maximum(r["y"], 0)

        if len(y) < 5:
            continue

        y = y / (np.linalg.norm(y) + 1e-12)

        y = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(y)),
            y
        )

        parts.append(y)

    if len(parts) == 0:
        return None

    template = np.mean(parts, axis=0)

    template /= np.linalg.norm(template) + 1e-12

    return template

# building template for every cluster
def compute_cluster_templates(raw_peak_results):

    cluster_ids = sorted(
        set(r["cluster_id"] for r in raw_peak_results)
    )

    templates = {}

    for cid in cluster_ids:

        template = build_cluster_template_shape(
            cid,
            raw_peak_results
        )

        if template is not None:
            templates[cid] = template

    return templates

# similarity between signal and template 
def normalized_cross_correlation(signal, template):

    signal = np.asarray(signal)
    template = np.asarray(template)

    signal = signal - np.mean(signal)
    template = template - np.mean(template)
    # cosine similarity matrix
    signal /= np.linalg.norm(signal) + 1e-12
    template /= np.linalg.norm(template) + 1e-12

    return np.correlate(
        signal,
        template,
        mode="full"
    )

# builds x- axis for correlation
def lag_axis(n, df):

    return np.arange(
        -n + 1,
        n
    ) * df

from scipy.ndimage import gaussian_filter1d
# looks for periods in the correlation signal
def extract_repetitions_robust(
    corr,
    freq_axis,
    f_min,
    f_max,
    top_k=5,
    smooth_sigma=2
):

    corr_smooth = gaussian_filter1d(
        corr,
        sigma=smooth_sigma
    )

    corr_smooth /= (
        np.max(np.abs(corr_smooth))
        + 1e-12
    )

    idx = np.argsort(
        corr_smooth
    )[-top_k:]

    positions = np.sort(
        freq_axis[idx]
    )

    positions = positions[
        (positions >= f_min)
        &
        (positions <= f_max)
    ]

    diffs = (
        np.diff(positions)
        if len(positions) > 1
        else []
    )

    return (
        positions,
        diffs,
        corr_smooth
    )

import numpy as np


def get_cluster_properties(
    raw_peak_results,
    frequency,
    activity=None,
    resonance_mask=None
):

    properties = {}


    for r in raw_peak_results:

        cid = r["cluster_id"]

        pos = r["peak_position"]

        amp = np.max(
            r["y"]
        )


        if activity is not None:

            idx = np.argmin(
                np.abs(frequency-pos)
            )

            peak_activity = activity[idx]

        else:

            peak_activity = 0



        if resonance_mask is not None:

            idx = np.argmin(
                np.abs(frequency-pos)
            )

            in_resonance = bool(
                resonance_mask[idx]
            )

        else:

            in_resonance = False



        if cid not in properties:

            properties[cid] = {
                "positions": [],
                "amplitudes": [],
                "activities": [],
                "resonance": []
            }



        properties[cid]["positions"].append(pos)

        properties[cid]["amplitudes"].append(amp)

        properties[cid]["activities"].append(
            peak_activity
        )

        properties[cid]["resonance"].append(
            in_resonance
        )



    for cid in properties:

        properties[cid]["position"] = np.mean(
            properties[cid]["positions"]
        )

        properties[cid]["amplitude"] = np.mean(
            properties[cid]["amplitudes"]
        )

        properties[cid]["activity"] = np.mean(
            properties[cid]["activities"]
        )

        properties[cid]["in_resonance"] = any(
            properties[cid]["resonance"]
        )


    return properties

# cosine like similarity between templates 
def compute_similarity_matrix(
    templates
):

    cluster_ids = list(
        templates.keys()
    )

    n = len(cluster_ids)

    sim = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            t1 = templates[
                cluster_ids[i]
            ]

            t2 = templates[
                cluster_ids[j]
            ]

            sim[i, j] = np.dot(
                t1,
                t2
            )

    return (
        cluster_ids,
        sim
    )

