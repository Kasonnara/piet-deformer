# -*- coding: utf-8 -*-
"""
    Piet-Modeler
    Shaping piet esotheric language programs into an image.
    Copyright (C) 2019  Kasonnara <kasonnara@laposte.net>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

over_fit_ratio = 1.175
debug = False

cmds2str = [[None, "push", "pop"],
            ["add", "substract", "multiply"],
            ["divide", "mod", "not"],
            ["greater", "pointer", "switch"],
            ["duplicate", "roll", "in(int)"],
            ["in(chr)", "out(int)", "out(chr)"]]

str2cmd = {cmds2str[i][j]: (i, j) for i in range(len(cmds2str)) for j in range(len(cmds2str[0])) if (i > 0 or j > 0)}


code2color = np.array([[[255, 192, 192], [255,   0,   0], [192,   0,   0]],
                        [[255, 255, 192], [255, 255,   0], [192, 192,   0]],
                        [[192, 255, 192], [  0, 255,   0], [  0, 192,   0]],
                        [[192, 255, 255], [  0, 255, 255], [  0, 192, 192]],
                        [[192, 192, 255], [  0,   0, 255], [  0,   0, 192]],
                        [[255, 192, 255], [255,   0, 255], [192,   0, 192]]], np.uint8)
"""code2color = (np.array([[[255, 192, 192, 255], [255,   0,   0, 255], [192,   0,   0, 255]],
                       [[255, 255, 192, 255], [255, 255,   0, 255], [192, 192,   0, 255]],
                       [[192, 255, 192, 255], [  0, 255,   0, 255], [  0, 192,   0, 255]],
                       [[192, 255, 255, 255], [  0, 255, 255, 255], [  0, 192, 192, 255]],
                       [[192, 192, 255, 255], [  0,   0, 255, 255], [  0,   0, 192, 255]],
                       [[255, 192, 255, 255], [255,   0, 255, 255], [192,   0, 192, 255]]]) / 255) # pour les png"""
"""code2color = np.array([[[255, 192, 192], [255,   0,   0], [192,   0,   0]],
                       [[255, 255, 192], [255, 255,   0], [192, 192,   0]],
                       [[192, 255, 192], [  0, 255,  0],  [  0, 192,   0]],
                       [[192, 255, 255], [  0, 255, 255], [  0, 192, 192]],
                       [[192, 192, 255], [  0,   0, 255], [  0,   0, 192]],
                       [[255, 192, 255], [255,   0, 255], [192,   0, 192]]]) / 255"""
colors2code = {"#FFC0C0":(0, 0), "#FFFFC0":(1, 0), "#C0FFC0":(2, 0), "#C0FFFF":(3, 0), "#C0C0FF":(4, 0), "#FFC0FF":(5, 0),
          "#FF0000":(0,1), "#FFFF00":(1,1), "#00FF00":(2,1), "#00FFFF":(3,1), "#0000FF":(4,1), "#FF00FF":(5,1),
          "#C00000":(0,2), "#C0C000":(1,2), "#00C000":(2,2), "#00C0C0":(3,2), "#0000C0":(4,2), "#C000C0":(5,2)}
# TODO change to integer codes by reversing code2color

color_white = np.array([255, 255, 255], np.uint8)
color_black = np.array([0, 0, 0], np.uint8)
model_color = np.array([200,200,200], np.uint8)
out_model_color = np.array([10,10,10], np.uint8)

start_color = code2color[(0,1)]
end_patern = [[None,        None,        color_black, None       ],
              [None,        None,        start_color, color_black],
              [color_black, start_color, start_color, color_black],
              [None,        color_black, color_black, None       ]]


def add_codes(previous_color, cmd_code):
    return (previous_color[0]+cmd_code[0])%6, (previous_color[1]+cmd_code[1])%3


class FixedVar:
    fixed = True
    variable = False


class DPdir:
    droite = (0, 1)
    gauche = (0, -1)
    haut = (-1, 0)
    bas = (1, 0)

    @staticmethod
    def next(dp):
        return dp[1], -dp[0]


class CCdir:
    gauche = -1
    droite = 1

class ConflictException(Exception):
    """Exception raised when attempting to put draw a pixel whit a conflicting pixel in the upper line"""
    pass

class ModelException(Exception):
    """Exception raised when attempting to put draw a pixel out of the model"""
    pass

class BorderException(Exception):
    """Exception levé quand on tente d'ajouter un bloc qui sort de l'écran"""
    pass
