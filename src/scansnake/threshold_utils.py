import numpy as np


def estimate_threshold(image):
    pixel_vals = image.flatten()
    # print(np.std(pixel_vals))
    if np.std(pixel_vals) < 10:
        return 0

    median_val = np.median(pixel_vals)

    def gen_thresh_scores():
        for thresh in range(10, int(median_val) + 1):
            mask = pixel_vals < thresh
            left = pixel_vals[mask]
            right = pixel_vals[~mask]
            if left.size / pixel_vals.size < 0.1:  # or (right.size < 3):
                continue

            left_mean = np.mean(left)
            left_std = np.std(left)

            right_mean = np.mean(right)
            right_std = np.std(right)
            score = abs(right_mean - left_mean) / (right_std + left_std)
            yield score, thresh

    _, thresh = max(gen_thresh_scores(), key=lambda x: x[0])
    return thresh


def threshold_image(image):
    bins = 10
    thresholds = np.zeros((bins, bins))
    bin_width = image.shape[1] // bins
    bin_height = image.shape[0] // bins
    bin_ixs_cols = [[i * bin_width, (i + 1) * bin_width] for i in range(bins)]
    bin_ixs_cols[-1][1] = None

    bin_ixs_rows = [[i * bin_height, (i + 1) * bin_height] for i in range(bins)]
    bin_ixs_rows[-1][1] = None
    for i in range(bins):
        for j in range(bins):
            # print(i, j)
            # print(bin_ixs_rows[i])
            # print(bin_ixs_cols[j])
            cols_ixs = slice(bin_ixs_cols[j][0], bin_ixs_cols[j][1])
            rows_ixs = slice(bin_ixs_rows[i][0], bin_ixs_rows[i][1])
            subimage = image[rows_ixs, cols_ixs]
            # print(image.shape)
            # print(subimage.shape)
            thresh = estimate_threshold(subimage)
            thresholds[i, j] = thresh
            # print('######')
    # print(thresholds)
    smoothed_thresholds = np.zeros((bins, bins))
    for i in range(1, bins - 1):
        for j in range(1, bins - 1):
            smoothed_thresholds[i, j] = np.median(
                thresholds[i - 1 : i + 2, j - 1 : j + 2].flatten()
            )
    # print(smoothed_thresholds)

    smoothed_thresholds[0, 0] = smoothed_thresholds[1, 1]
    smoothed_thresholds[bins - 1, 0] = smoothed_thresholds[bins - 2, 1]
    smoothed_thresholds[0, bins - 1] = smoothed_thresholds[1, bins - 2]
    smoothed_thresholds[bins - 1, bins - 1] = smoothed_thresholds[bins - 2, bins - 2]
    # print(smoothed_thresholds)
    for i in range(1, bins - 1):
        smoothed_thresholds[0, i] = smoothed_thresholds[1, i]
        smoothed_thresholds[i, 0] = smoothed_thresholds[i, 1]
        smoothed_thresholds[i, bins - 1] = smoothed_thresholds[i, bins - 2]
        smoothed_thresholds[bins - 1, i] = smoothed_thresholds[bins - 2, i]
    # print(smoothed_thresholds)

    out = np.zeros(image.shape, np.uint8)
    out += 255
    for i in range(bins):
        for j in range(bins):
            # print(i, j)
            # print(bin_ixs_rows[i])
            # print(bin_ixs_cols[j])
            cols_ixs = slice(bin_ixs_cols[j][0], bin_ixs_cols[j][1])
            rows_ixs = slice(bin_ixs_rows[i][0], bin_ixs_rows[i][1])
            subimage = image[rows_ixs, cols_ixs]
            # print(image.shape)
            # print(subimage.shape)
            # thresh = estimate_threshold(subimage)
            thresh = smoothed_thresholds[i, j]
            mask = subimage < thresh
            mask = mask.reshape(subimage.shape)
            subout = np.zeros(mask.shape, np.uint8)
            subout += 255
            subout[mask] = 0
            # print(mask)
            # print(mask.shape)
            # print(subout)
            # print(subout.shape)
            out[rows_ixs, cols_ixs] = subout
            # print('######')
    return out
