# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License

from typing import List

import numpy as np

from core.basic import PietColors
from instruction_generation.static_message import BasicMessageDisplay


def line_modeler(program: BasicMessageDisplay) -> np.ndarray:
    # Init the image
    out_img = np.zeros((1, program.estimated_pixel_lenght() + 1, 3))
    # Iterate over instructions
    index = 0
    current_code = PietColors.RED.color_code
    for instruction, codel_size in program.instructions:
        out_img[0, index: index + codel_size] = PietColors.code2color[current_code].numpy_color
        current_code = current_code + instruction.color_code
        index += codel_size
    # Put the last pixel
    out_img[0, -1] = PietColors.code2color[current_code].numpy_color

    #print(out_img)
    #Image.imsave("linarized_code.png", out_img)
    return out_img

