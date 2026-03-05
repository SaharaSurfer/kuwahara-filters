import cv2
import numpy as np


def classic_kuwahara(image, radius: int = 3):
    """
    The Kuwahara filter works on a window divided into 4 overlapping
    subwindows. In each subwindow, the mean and variance are computed.
    The output value (located at the center of the window) is set to
    the mean of the subwindow with the smallest variance.

    Example for a 5x5 pixel window:

          ( a  a  ab   b  b)
          ( a  a  ab   b  b)
          (ac ac abcd bd bd)
          ( c  c  cd   d  d)
          ( c  c  cd   d  d)

    """
    if radius == 0:
        return image.copy()

    subwindow_size = (radius + 1, radius + 1)
    subwindow_anchors = [
        (radius, radius),
        (0, radius),
        (radius, 0),
        (0, 0),
    ]

    H, W = image.shape[:2]
    means = np.zeros((4, H, W, 3), dtype=np.float32)
    variances = np.zeros((4, H, W), dtype=np.float32)

    img_float = image.astype(np.float32) / 255.0
    img_float_sq = img_float**2

    for i, anchor in enumerate(subwindow_anchors):
        mean_rgb = cv2.boxFilter(
            src=img_float,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        )

        mean_rgb_sq = cv2.boxFilter(
            src=img_float_sq,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        )

        var_per_channel = np.clip(mean_rgb_sq - mean_rgb**2, 0.0, None)
        total_var = np.sum(var_per_channel, axis=2)

        means[i] = mean_rgb
        variances[i] = total_var

    min_var_idx = np.argmin(variances, axis=0)

    yy, xx = np.indices((H, W))
    result = means[min_var_idx, yy, xx]

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def classic_kuwahara_flawed(image, radius: int = 3):
    if radius == 0:
        return image.copy()

    subwindow_size = (radius + 1, radius + 1)
    subwindow_anchors = [
        (radius, radius),
        (0, radius),
        (radius, 0),
        (0, 0),
    ]

    H, W = image.shape[:2]
    means = np.zeros((4, H, W, 3), dtype=np.float32)
    variances = np.zeros((4, H, W, 3), dtype=np.float32)

    img_float = image.astype(np.float32) / 255.0
    img_float_sq = img_float**2

    for i, anchor in enumerate(subwindow_anchors):
        mean_rgb = cv2.boxFilter(
            src=img_float,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        )

        mean_rgb_sq = cv2.boxFilter(
            src=img_float_sq,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        )

        var_per_channel = np.clip(mean_rgb_sq - mean_rgb**2, 0.0, None)

        means[i] = mean_rgb
        variances[i] = var_per_channel

    min_var_idx = np.argmin(variances, axis=0)

    yy, xx = np.indices((H, W))
    result = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        result[:, :, c] = means[min_var_idx[:, :, c], yy, xx, c]

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
