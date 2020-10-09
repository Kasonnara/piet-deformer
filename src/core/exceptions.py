# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
Piet modeler inner working exceptions

As a user you should never see them as they must be caught by the Piet modeler process as a common behaviour.
"""


class ColorConflict(Exception):
    """Exception raised when attempting to draw a pixel which is touching another bloc of the same color"""


class OutOfMask(Exception):
    """Exception raised when attempting to draw a pixel out of the mask"""
    def __init__(self, recommended_offset: int = 1, *msgs: object) -> None:
        super().__init__(*msgs)
        self.recommended_offset = recommended_offset


class OutOfBound(Exception):
    """Exception raised when attempting to draw a bloc out of the image"""
    def __init__(self, expected_space=None):
        self.expected_space = expected_space


class NotEnoughSpace(Exception):
    """Exception raised when we hit the bottom of the image while still needing to draw some instructions"""
    pass
