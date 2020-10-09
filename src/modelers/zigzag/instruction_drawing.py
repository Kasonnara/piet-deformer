# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
Low level function for checking/drawing a single pixel/instruction by the zigzag modeler
"""
from typing import List, Optional

import numpy

from core.types import PietInstruction, PietColors, State, PietInstructionTypes as PIT, DPDir
from core.exceptions import ColorConflict, OutOfMask, OutOfBound
from core.utils import xy_line_range
from core.types import Mode


def draw_pixel(out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, x: int, y: int, codel_color: PietColors, mode=Mode.NORMAL):
    """
    Draw a pixel if dry is not set, else check if there is no ColorConflict nor OutOfMask
    """
    if mode.disable_drawing:
        #if x < 0 or out_image.shape[0] <= x or y < 0 or out_image.shape[1] <= y
        #    raise OutOfBound()
        # Check color conflict
        if state.x > 0:  # No conflict if we are drawing the first line
            if x >= state.x:  # No conflict if we are moving down
                if codel_color is not PietColors.WHITE:  # White pixel does not make conflicts
                    if out_image[x - 1][y] is codel_color:
                        raise ColorConflict()
        if not mode.ignore_mask:
            # Check we are on the mask
            if not mask[x, y]:
                raise OutOfMask()  # FIXME set recommended offset value
    else:
        # Regular paint
        out_image[x][y] = codel_color


def draw_instruction(out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS, instruction: PietInstruction, mode=Mode.NORMAL):
    """
    Draw the given instruction, and update draw state (state.x, state.y, state.current_code and state.dp).

    :param out_image: List[List[Optional[PietColors]]], the image to draw on.
    :param mask: numpy.ndarray[bool], the mask whose shape we are trying to match.
    :param state:  State, the current state of the Piet Interpreter when entering this bloc.
    :param instruction: PietInstruction, the instruction to draw.
    :param force: bool, If True, do not automatically run check_instruction_drawable. Used this when drawing
                        patterns of multiple codels on which you already run the checks.
    :raise OutOfBound, OutOfMask, ColorConflict: see check_instruction_drawable.
    """
    assert (state.dp.x == 0) or (state.dp.x == 1 and (instruction.instruction_type is PIT.POINTER or instruction.instruction_type is PIT.NO_OP)), "Wrong use of directional pointer. ZigZag modeler should only have none zero dp.x when for a Pointer or no op instruction"

    # Check if the codel can be drawn
    if mode.disable_drawing:
        # Check for margins (This shouldn't be necessary, if the mask have margins, the OutOfMask should trigger first)
        if not 0 <= state.y + instruction.value * state.dp.y < mask.shape[1]:
            raise OutOfBound(instruction.value + 1)
    if mode is Mode.NORMAL:
        # Automatically run dry mode
        draw_instruction(out_image, mask, state, TURN_MARGINS, instruction, mode=Mode.DRY)

    # Draw the codel
    codel_color = PietColors.code2color[state.current_code] if instruction.instruction_type is not PIT.NO_OP else PietColors.WHITE
    x, y = state.x, state.y
    for x, y in xy_line_range(state.x, state.y, instruction.value, state.dp, skip_first=True):
        draw_pixel(out_image, mask, state, x, y, codel_color, mode=mode)

    # Update state
    x, y = x + state.dp.x, y + state.dp.y
    if instruction.instruction_type is PIT.NO_OP:
        assert instruction.value >= 2
        # Use the modeler hint if provided
        if instruction.modeler_hints is not None:
            next_color_code = instruction.modeler_hints
        else:
            next_color_code = state.current_code  # TODO Here we may make better chose
    else:
        next_color_code = state.current_code + instruction.instruction_type.color_code

    if not mode.disable_state_change:
        state.x, state.y = x, y
        if instruction.instruction_type is PIT.POINTER and instruction.modeler_hints is not None:
            # Pointer instruction can hint the expected direction/rotation:
            if isinstance(instruction.modeler_hints, DPDir):
                state.dp = instruction.modeler_hints
            else:
                for _ in range(instruction.modeler_hints):
                    state.dp = state.dp.next
        # Update the next color to draw to get the instruction
        state.current_code = next_color_code

    # Draw the first pixel of the next codel
    draw_pixel(out_image, mask, state, x, y, PietColors.code2color[next_color_code], mode=mode)
