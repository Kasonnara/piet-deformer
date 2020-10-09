# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
Utility functions
"""
from typing import Tuple, List, Optional

import numpy
import numpy as np
import PIL.Image as PILImage

from core.types import DPDir


def to_mask(
        image:  PILImage.Image,
        use_alpha=True,
        threshold=128,
        invert=False,
        margins: Tuple[int, int, int, int]=(0,0,0,0),
        resize_to: Tuple[int, int]=None,
        ) -> np.ndarray:
    """
    Convert the given image into a 2 dimensional array of booleans usable as mask.

    :param image: numpy.ndarray, a 3 dimensional array of int to use as mask.
    :param use_alpha: bool, if True use the alpha canal as mask on image with alpha enable.
    :param threshold: int, the threshold above which the pixel is in the mask.
    :param invert: bool, invert the threshold
    :param margins: tuple of 4 integer, the process will ensure that there is at least this number of False lines on
        each side of the mask. Values respectively indicates margin on (top, left, bottom, right)

    :return: numpy.ndarray[bool], a 2 dimensional array of booleans.
    """
    if resize_to:
        image = image.copy()
        image.thumbnail(resize_to, resample=PILImage.ANTIALIAS)

    # Convert to numpy
    numpy_image = np.array(image)
    # If necessary convert values integer
    if numpy_image.dtype == np.float64:
        numpy_image = (numpy_image * 256).astype(np.uint)

    if use_alpha:
        # Use the alpha canal as the mask
        grey_image = numpy_image[:, :, 3]
    else:
        # Use the mean of the colors as the mask ~= luminosity
        grey_image = np.mean(numpy_image[:, :, :3], axis=2)

    # Convert to boolean
    if invert:
        mask = grey_image < threshold
    else:
        mask = grey_image > threshold

    # Ensure there are empty margins on the borders of the image
    for top_offset in range(margins[0]):
        if any(mask[top_offset, :]):
            added_lines = np.array([[False for _ in range(mask.shape[1])] for _ in range(margins[0] - top_offset)])
            mask = np.concatenate((added_lines, mask))
            break
    for bottom_offset in range(-1, -margins[2]-1, -1):
        if any(mask[bottom_offset, :]):
            added_lines = np.array([[False for _ in range(mask.shape[1])] for _ in range(margins[2] - (-bottom_offset + 1))])
            mask = np.concatenate((mask, added_lines))
            break
    for left_offset in range(margins[1]):
        if any(mask[:, left_offset]):
            added_lines = np.array([[False for _ in range(margins[1] - left_offset)] for _ in range(mask.shape[0])])
            mask = np.concatenate((added_lines, mask), axis=1)
            break
    for right_offset in range(-1, -margins[3] - 1, -1):
        if any(mask[:, right_offset]):
            added_lines = np.array([[False for _ in range(margins[3] - (-right_offset + 1))] for _ in range(mask.shape[0])])
            mask = np.concatenate((mask, added_lines), axis=1)
            break

    return mask


def xy_line_range(x0: int, y0:int, delta: int, direction: DPDir, skip_first=False):
    """An iterator returning the (x,y) coordinates of cells on a line from (x0, y0) of length delta in the DP direction."""
    for k in range(1 if skip_first else 0, delta):
        yield x0 + k * direction.x, y0 + k * direction.y


def xy_line_slice(x0: int, y0:int, delta: int, direction: DPDir, skip_first=False):
    """A 2D numpy array slice that select a line from (x0, y0) of length delta in the DP direction."""
    return (
        slice(x0 + (direction.x if skip_first else 0), x0 + delta * direction.x, direction.x) if direction.x else x0,
        slice(y0 + (direction.y if skip_first else 0), y0 + delta * direction.y, direction.y) if direction.y else y0,
    )


def to_image(out_image: List[List[Optional['PietColors']]], mask: numpy.ndarray,
             inner_color = None, outer_color=None) -> PILImage.Image:
    """Convert the array of colors used in modelers process into a proper PIL image"""

    inner_color = inner_color if inner_color is not None else (outer_color if outer_color is not None else [255, 255, 255])
    outer_color = outer_color if outer_color is not None else inner_color
    out_numpy = numpy.array([
        [color.numpy_color if color is not None else (inner_color if mask[x][y] else outer_color)
         for y, color in enumerate(line)]
        for x, line in enumerate(out_image)
        ])
    return PILImage.fromarray(out_numpy.astype(np.uint8))
