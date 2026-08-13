import numpy as np
def build_peak_space(
    frequency,
    amplitude,
    peaks,
    activity
):

    peak_freqs = np.asarray(
        frequency[peaks],
        dtype=float
    )

    peak_amps = np.asarray(
        amplitude[peaks],
        dtype=float
    )

    peak_activity = np.asarray(
        activity[peaks],
        dtype=float
    )


    order = np.argsort(
        peak_freqs
    )


    return (
        peaks[order],
        peak_freqs[order],
        peak_amps[order],
        peak_activity[order]
    )

def cluster_peaks_by_frequency(
    peaks,
    frequency,
    amplitude,
    peak_to_group,
    cluster_gap_threshold=1.0e5,
    amp_ratio_threshold=0.6,
    valley_threshold=0.4,
    group_weight=0.5
):

    if len(peaks) == 0:
        return []


    peaks = np.asarray(
        peaks,
        dtype=int
    )


    peaks = peaks[
        np.argsort(
            frequency[peaks]
        )
    ]


    clusters = []

    current = [
        peaks[0]
    ]



    for i in range(1, len(peaks)):


        p_prev = current[-1]
        p_curr = peaks[i]


        gap = (
            frequency[p_curr]
            -
            frequency[p_prev]
        )

        # Group Information

        g_prev = peak_to_group.get(
            p_prev,
            None
        )

        g_curr = peak_to_group.get(
            p_curr,
            None
        )


        same_group = (
            g_prev is not None
            and
            g_prev == g_curr
        )

        # Amplitude
        
        amp_ratio = (
            amplitude[p_curr]
            /
            (amplitude[p_prev]+1e-12)
        )


        similar_height = (

            amp_ratio > amp_ratio_threshold

            and

            amp_ratio < 1/amp_ratio_threshold

        )
        # Valley

        window = amplitude[
            p_prev:p_curr+1
        ]


        valley_min = np.min(
            window
        )


        peak_min = min(
            amplitude[p_prev],
            amplitude[p_curr]
        )


        valley_ratio = (

            valley_min
            /
            (peak_min+1e-12)

        )


        deep_valley = (
            valley_ratio < valley_threshold
        )



        # ----------------------------------
        # Score
        # ----------------------------------

        score = 0


        if gap <= cluster_gap_threshold:
            score += 1


        if same_group:
            score += group_weight

        else:
            score -= group_weight



        if similar_height:
            score += 0.5


        if not deep_valley:
            score += 0.5



        if score >= 1.0:

            current.append(
                p_curr
            )

        else:

            clusters.append(
                current
            )

            current = [
                p_curr
            ]



    clusters.append(
        current
    )


    print(
        "Clusters:",
        len(clusters)
    )

    print(
        "Cluster sizes:",
        [
            len(c)
            for c in clusters
        ]
    )


    return clusters

# left Peak in the cluster and right peak in the cluster for the window 
# width and flank factor and if the window collapses a fallback of 5 indices
def extract_cluster_windows(frequency, clusters, flank_factor=0.2):

    windows = []
    n = len(frequency)

    for cluster in clusters:

        if len(cluster) == 0:
            continue

        # single peaks are also allowed!
        left_idx = cluster[0]
        right_idx = cluster[-1]

        f_left = frequency[left_idx]
        f_right = frequency[right_idx]

        width = f_right - f_left

        if width == 0:
            width = (frequency[1] - frequency[0]) * 10  

        f_start = f_left - flank_factor * width
        f_end   = f_right + flank_factor * width

        start_idx = np.searchsorted(frequency, f_start)
        end_idx   = np.searchsorted(frequency, f_end)

        # Clipping
        start_idx = max(0, start_idx)
        end_idx = min(n - 1, end_idx)

    
        if end_idx <= start_idx:
            center = left_idx
            start_idx = max(0, center - 5)
            end_idx = min(n - 1, center + 5)

        windows.append((start_idx, end_idx))

    return windows

# using the derivative to refine the windows -> end if derivative gets small
# patience how many points under the threshold are okay
# max_width: how wide the window can be max
def refine_window_with_derivative(
    amplitude,
    peak_idx,
    gradient,
    max_width=80,
    rel_slope=0.3,
    patience=3
):

    grad = gradient
    peak_grad = grad[peak_idx]

    threshold = rel_slope * peak_grad

    left = peak_idx
    right = peak_idx

    # links
    below_count = 0
    while left > 1:
        if grad[left] < threshold:
            below_count += 1
            if below_count >= patience:
                break
        else:
            below_count = 0

        if peak_idx - left > max_width:
            break

        left -= 1

    # rechts
    below_count = 0
    while right < len(amplitude) - 2:
        if grad[right] < threshold:
            below_count += 1
            if below_count >= patience:
                break
        else:
            below_count = 0

        if right - peak_idx > max_width:
            break

        right += 1

    return left, right

# frequency width half maximum
# not enough points, return 0 , linear interpolation 
def compute_fwhm(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    half_max = np.max(y) / 2

    above = np.where(y >= half_max)[0]

    if len(above) < 2:
        return 0.0

    i_left = above[0]
    i_right = above[-1]

    if i_left == 0:
        x_left = x[0]
    else:
        x1, x2 = x[i_left - 1], x[i_left]
        y1, y2 = y[i_left - 1], y[i_left]

        x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    if i_right == len(y) - 1:
        x_right = x[-1]
    else:
        x1, x2 = x[i_right], x[i_right + 1]
        y1, y2 = y[i_right], y[i_right + 1]

        x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    return x_right - x_left

# position error for the peaks with fwhm
def estimate_position_error(fwhm):

    return fwhm / (2 * np.sqrt(2 * np.log(2)))

from scipy.signal import find_peaks
# single peak or double peak? 
# merging if more than two peaks and weighted mean value for the position / the peak
def analyze_peak_shape_raw(x, y):

    peaks, _ = find_peaks(
        y,
        prominence=np.max(y) * 0.1
    )

    if len(peaks) == 1:

        idx = peaks[0]

        peak_pos = x[idx]
        peak_amp = y[idx]

        fwhm = compute_fwhm(x, y)
        pos_err = estimate_position_error(fwhm)

        return {
            "type": "single",
            "peak_position": peak_pos,
            "amplitude": peak_amp,
            "fwhm": fwhm,
            "position_error": pos_err
        }

    elif len(peaks) >= 2:

        peak_heights = y[peaks]

        top2_idx = np.argsort(peak_heights)[-2:]

        p1, p2 = peaks[top2_idx]

        x1, x2 = x[p1], x[p2]
        y1, y2 = y[p1], y[p2]

        pos = (x1*y1 + x2*y2) / (y1 + y2)

        fwhm = compute_fwhm(x, y)

        pos_err = estimate_position_error(fwhm)

        return {
            "type": "double",
            "peak_position": pos,
            "peaks_positions": (x1, x2),
            "amplitudes": (y1, y2),
            "fwhm": fwhm,
            "position_error": pos_err
        }

    return None 

def extract_peak_segments_raw(
    frequency,
    amplitude,
    clusters,
    peak_to_group
):

    gradient = np.abs(np.gradient(amplitude))

    coarse_windows = extract_cluster_windows(
        frequency,
        clusters
    )

    results = []

    for i, cluster in enumerate(clusters):

        if len(cluster) == 0:
            continue

        if i >= len(coarse_windows):
            continue

        left_c, right_c = coarse_windows[i]

        # Center = around tallest peak

        peak_center = cluster[
            np.argmax(
                amplitude[cluster]
            )
        ]

        left, right = refine_window_with_derivative(
            amplitude,
            peak_center,
            gradient
        )

        if right <= left:
            left, right = left_c, right_c

        left = max(left, peak_center - 30)
        right = min(right, peak_center + 30)

        x_win = frequency[left:right]
        y_win = amplitude[left:right]

        if len(x_win) < 5:
            continue

        # intensity profile

        y_raw = y_win - np.min(y_win)

        y_raw = np.maximum(y_raw, 0)

        #shape normalised profile
        y_shape = y_raw.copy()

        if np.max(y_shape) > 0:
            y_shape = y_shape / np.max(y_shape)


        analysis = analyze_peak_shape_raw(
            x_win,
            y_shape
        )


        peak_idx = np.argmax(y_shape)

        peak_pos = x_win[peak_idx]

        peak_height = y_shape[peak_idx]


        area = np.trapezoid(
            y_raw,
            x_win
        )


        results.append({

            "cluster_id": i,

            "group_id": peak_to_group.get(
                peak_center,
                None
            ),

            "cluster_peaks": cluster,

            "window": (
                left,
                right
            ),

            "x": x_win,

            "y": y_shape,


            "y_raw": y_raw,


            "peak_position": peak_pos,

            "peak_height": peak_height,

            "area": area,

            "analysis": analysis

        })

    return results

