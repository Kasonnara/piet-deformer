# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License

"""
This file contain basic data structures for Piet language concepts.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

over_fit_ratio = 1.175

HUE_LENGHT = 6
LUMINOSITY_LENGHT = 3


class ColorCode(Enum):
    """Enumerate piet all possible Hue/Luminosity codel codes"""
    RL = (0, 0)
    R  = (0, 1)
    RD = (0, 2)
    YL = (1, 0)
    Y  = (1, 1)
    YD = (1, 2)
    GL = (2, 0)
    G  = (2, 1)
    GD = (2, 2)
    CL = (3, 0)
    C  = (3, 1)
    CD = (3, 2)
    BL = (4, 0)
    B  = (4, 1)
    BD = (4, 2)
    ML = (5, 0)
    M  = (5, 1)
    MD = (5, 2)

    def __init__(self, hue, luminosity):
        self.hue: int = hue
        self.luminosity: int = luminosity

    def __add__(self, other: 'ColorCode') -> 'ColorCode':
        return ColorCode((
            (self.hue + other.hue) % HUE_LENGHT,
            (self.luminosity + other.luminosity) % LUMINOSITY_LENGHT,
            ))

    def rotate(self):
        return ColorCode((self.hue            if self.luminosity < 2 else (self.hue + 1) % 6,
                          self.luminosity + 1 if self.luminosity < 2 else 0))


class PietInstructionTypes(Enum):
    """
    An enumeration of all piet instructions.

    Not really used as is. It's mostly there to put names on things in an organized fashion, and next generate from it
    arrays and dictionary to efficiently access data.
    """
    _IMPOSSIBLE = ColorCode.RL
    PUSH        = ColorCode.R
    POP         = ColorCode.RD
    ADD         = ColorCode.YL
    SUBTRACT    = ColorCode.Y
    MULTIPLY    = ColorCode.YD
    DIVIDE      = ColorCode.GL
    MOD         = ColorCode.G
    NOT         = ColorCode.GD
    GREATER     = ColorCode.CL
    POINTER     = ColorCode.C
    SWITCH      = ColorCode.CD
    DUPLICATE   = ColorCode.BL
    ROLL        = ColorCode.B
    IN_INT      = ColorCode.BD
    IN_CHR      = ColorCode.ML
    OUT_INT     = ColorCode.M
    OUT_CHR     = ColorCode.MD
    NO_OP       = None

    def __init__(self, color_code: ColorCode):
        self.color_code = color_code
        self.instruction: callable = None
        self.predictive_analysis_instruction: callable = None

    @property
    def is_codel_size_important(self):
        return self is self.PUSH


@dataclass
class PietInstruction:
    instruction_type: PietInstructionTypes
    value: int = 1
    """Codel value: relevant only for PUSH instruction, but can be used to manipulate codel size, 
    especially for noop instruction. Must be >0 (and >1 for NO_OP instructions)"""
    modeler_hints: object = None
    """Extra hints for the modeler:
    - for POINTER: it can be an int to indicate what fixed value should be on top of the stack, or directly a DPDir to set.
    - for NO_OP: it can be a ColorCode to use as the new current color code.
    - for SWITCH: not implemented yet TODO
    """
    def __repr__(self):
        return "{}({}){}".format(self.instruction_type.name, self.value, "[{}]".format(self.modeler_hints) if self.modeler_hints is not None else "")


class PietColors(Enum):
    """
     An enumeration of valid piet colors.

     Not really used as is. It's mostly there to put names on things in an organized fashion, and next generate from it
     arrays and dictionary to efficiently access data.
     """
    LIGHT_RED       = ColorCode.RL, "FFC0C0"
    RED             = ColorCode.R , "FF0000"
    DARK_RED        = ColorCode.RD, "C00000"

    LIGHT_YELLOW    = ColorCode.YL, "FFFFC0"
    YELLOW          = ColorCode.Y , "FFFF00"
    DARK_YELLOW     = ColorCode.YD, "C0C000"

    LIGHT_GREEN     = ColorCode.GL, "C0FFC0"
    GREEN           = ColorCode.G , "00FF00"
    DARK_GREEN      = ColorCode.GD, "00C000"

    LIGHT_CYAN      = ColorCode.CL, "C0FFFF"
    CYAN            = ColorCode.C , "00FFFF"
    DARK_CYAN       = ColorCode.CD, "00C0C0"

    LIGHT_BLUE      = ColorCode.BL, "C0C0FF"
    BLUE            = ColorCode.B , "0000FF"
    DARK_BLUE       = ColorCode.BD, "0000C0"

    LIGHT_MAGENTA   = ColorCode.ML, "FFC0FF"
    MAGENTA         = ColorCode.M , "FF00FF"
    DARK_MAGENTA    = ColorCode.MD, "C000C0"

    BLACK = None, "000000"
    WHITE = None, "FFFFFF"

    def __init__(self, color_code: ColorCode, html_color):
        self.color_code = color_code
        self.html_color = html_color
        # Convert html color code into numpy array
        self.numpy_color = np.array([int(html_color[0:2], 16), int(html_color[2:4], 16), int(html_color[4:6], 16)])

    # ----- CAST FUNCTIONS -----

    @classmethod
    def html2tuple(cls, html_color_code: str) -> Tuple[int, int, int]:
        """Convert string html color codes into the corresponding triplet"""
        return cls.html2color[html_color_code].tuple_color
        # Unless you need to manipulate colors that are not PietColor, 'PietColors.html2color[html_color].tuple_color'
        # is more efficient and lead to more code reuse. Else you can use this slower yet more flexible code:
        #return (int(html_color_code[0:2], 16),
        #        int(html_color_code[2:4], 16),
        #        int(html_color_code[4:6], 16))

    @classmethod
    def html2numpy(cls, html_color_code: str) -> np.ndarray:
        """Convert string html color codes into the corresponding numpy array"""
        return cls.html2color[html_color_code].numpy_color
        # Unless you need to manipulate colors that are not PietColor, 'PietColors.html2color[html_color].numpy_color'
        # is more efficient and lead to more code reuse. Else you can use this slower yet more flexible code:
        #return np.array(list(cls.html2tuple(html_color_code)))

    @classmethod
    def numpy2color(cls, numpy_color: np.ndarray) -> 'PietColors':
        """
        Convert numpy array pixel into the corresponding HTML color code
        :param numpy_color: ndarray[int]: a 1-dimension numpy array of length 3 containing the RGB values of the
               pixel as integer in range [0;255]
        """
        return cls.tuple2color[tuple(numpy_color)]

    @classmethod
    def numpy2html(cls, numpy_color: np.ndarray) -> str:
        """
        Convert numpy array pixel into the corresponding HTML color code
        :param numpy_color: ndarray[int]: a 1-dimension numpy array of length 3 or 4 containing the RGB values of the
               pixel as integer in range [0;255]
        """
        return cls.numpy2color(numpy_color).html_color
        # (unless you need to manipulate colors that are not PietColor, 'PietColors.numpy2color(numpy_color).html_color'
        # is more efficient and lead to more code reuse. Else you can use this slower yet more flexible code:
        # return "{0:02x}{1:02x}{2:02x}".format(*numpy_color)


# Generate a matrix for matching code to instruction more efficiently
PietInstructionTypes.code2instr = [[None for _ in range(LUMINOSITY_LENGHT)] for _ in range(HUE_LENGHT)]
"""Array to convert PietCode into the corresponding piet instruction"""
for instr in PietInstructionTypes:
    if instr.color_code is not None:
        PietInstructionTypes.code2instr[instr.color_code.hue][instr.color_code.luminosity] = instr

# Generate a dictionary to map html codes and tuples to their corresponding color object
PietColors.code2color = {color.color_code: color for color in PietColors}
"""Dict to convert PietCode into the corresponding PietColors"""
PietColors.html2color = {color.html_color: color for color in PietColors}
"""Dict to convert string HTML color code into the corresponding PietColors"""
PietColors.tuple2color = {tuple(color.numpy_color): color for color in PietColors}
"""Dict to convert int(0-255) RGB triplet into the corresponding PietColors"""


class DPDir(Enum):
    """Enumeration of the possible Directional Pointer (DP) states"""
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)
    UP = (-1, 0)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.next = None


DPDir.RIGHT.next = DPDir.DOWN
DPDir.DOWN.next = DPDir.LEFT
DPDir.LEFT.next = DPDir.UP
DPDir.UP.next = DPDir.RIGHT


class CCDir(Enum):
    """
    Enumeration of the possible Codel Chooser (CC) states
    """
    LEFT = -1
    RIGHT = 1


CCDir.LEFT.invert = CCDir.RIGHT
CCDir.RIGHT.invert = CCDir.LEFT


@dataclass
class State:
    """
    A state of piet interpreter.

    Note: this is a mutable object made to be modified as the instructions are read. So mind to use copy() and recopy()
    if you need to.
    """
    x: int
    y: int
    dp: DPDir
    cc: CCDir
    current_code: ColorCode

    def copy(self):
        """Create a new state instance identical to self"""
        return type(self)(self.x, self.y, self.dp, self.cc, self.current_code)

    def __repr__(self):
        return "[x={} y={} dp={} cc={} color={}]".format(self.x, self.y, self.dp.name, self.cc.name, self.current_code.name)

    def recopy(self, other: 'State'):
        """Copy back the state of other to self"""
        # self.__dict__.update(other.__dict__)
        self.x, self.y, self.dp, self.cc, self.current_code = other.x, other.y, other.dp, other.cc, other.current_code


class Mode(Enum):
    """
    Piet code generation is often done in a try except way, but in many cases we need to need to process multiple
    instructions before encountering an error. So instead of trying to rewind all changes it's less error prone to run
    the sequence of instruction in dry mode where we check for conflict but without drawing anything.

    So relevant function provide a Mode type parameter for controling that.
    The mode contain 3 information:
    - is drawing disabled: if True we run in dry mode, checks must be done but nothing will be done
    - is state freezed: if True, state will be left untouched as well
    - is mask ignored: if True, the drawing function is allowed to ignore the shape mask, this is used by flow control
      instructions that must be drawn by any means necessary (like turns in the zigzag modeler).
    """

    NORMAL = (False, False, False)
    """Normal call of the function, the function may call itself recursively in DRY mode."""
    DRY = (True, True, False)
    """In DRY mode the function will run all sorts of checks but without modifying anything."""
    PRIORITIZED_DRY = (True, True, True)
    """Like DRY mode but ignore mask."""
    DRY_VIRTUALIZED = (True, False, False)
    """Like in DRY mode the function will not draw anything and run checks but will still modify the PietState. 
    This can be used to make a dry run over a whole sequence of instructions. But in this case, mind to 
    provide a virtual copy of the state, not the main one"""
    PRIORITIZED_VIRTUALIZED = (True, False, True)
    """Same as DRY_VIRTUALIZED but when you also need to ignore the mask"""
    FORCE = (False, False, True)
    """No checks are done, we assume everything has already been verified"""

    def __init__(self, disable_drawing: bool, disable_state_change: bool, ignore_mask: bool):
        self.disable_drawing = disable_drawing
        self.disable_state_change = disable_state_change
        self.ignore_mask = ignore_mask

    def virtualized(self):
        return Mode((self.disable_drawing, False, self.ignore_mask))

    def dry(self):
        return Mode((True, True, self.ignore_mask))
