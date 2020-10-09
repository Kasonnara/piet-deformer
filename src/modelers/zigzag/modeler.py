# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License

"""
Macroscopic level functions of the zigzag modeler
"""

import itertools
from typing import List, Optional, Union

import numpy
import numpy as np
import PIL.Image as PILImage

from core.types import PietInstructionTypes as PIT, PietColors, CCDir, DPDir, State, \
    PietInstruction
from core.exceptions import OutOfBound, ColorConflict, OutOfMask, NotEnoughSpace
from core.utils import to_mask, to_image
from core.types import Mode
from instruction_generation.static_message import BasicMessageDisplay
from modelers.zigzag.instruction_drawing import draw_pixel, draw_instruction
from modelers.zigzag.instructions_blocs_drawing import InstructionBloc, Turn, End


def _init(mask: np.ndarray, initial_color: PietColors):
    state = State(0, 0, DPDir.RIGHT, CCDir.RIGHT, initial_color.color_code)
    empty_piet_image: List[List[Optional[PietColors]]] = [[None for _ in range(mask.shape[1])]
                                                          for _ in range(mask.shape[0])]
    # empty_piet_image: np.ndarray = numpy.array(empty_piet_image)  # cast to numpy array
    # Draw the first pixel
    empty_piet_image[0][0] = initial_color

    TURN_MARGINS = (0 + Turn.OTHER_TURN_SIZE + Turn.offset_security + 5 , len(empty_piet_image[0]) - (Turn.RIGHT_RIGHT_TURN_SIZE + Turn.offset_security + 5))
    # FIXME: the "+5" is for allowing more space for a potential End bloc close to the border
    # FIXME: in theory as long as assure there is enough unmasked space on the sides by extending image size, we
    #  shouldn't need TURN_MARGINS checks, mask checks should be sufficient.
    return state, empty_piet_image, TURN_MARGINS


def _draw_auto_offset(out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS, target: Union[PietInstruction, 'InstructionBloc'], *args, ignore_mask=False, **kwargs):
    """
    Draw the given PietInstruction or InstructionBloc while automatically solving OutOfMask and
    ColorConflict exceptions that may be raised by adding NO_OP spaces, if this doesn't solve the
    problem (e.g. if we run out of space on the current line) an OutOfBound will finally be raised and it
    will pass though.

    :param target: Union[PietInstruction, 'InstructionBloc']

    :raise OutOfBound: raised by draw_instruction if adding blank space doesn't solve the problem, we must
      skip to the next line.
    :raise NotEnoughSpace: raised by _draw_turn if we hit the bottom of the image.
    """
    # TODO, currently we reset the color cycle to zero each time, maybe there is better way to test color
    #   that lead to less conflict in general.

    # TODO: Currently went adding offset we may skip pixels allowed by the mask, it may be possible to better match
    #  the mask by trying to fill those pixels with dummy codels with no effect.

    # TODO: currently instruction often are bi PUSH instruction, so it may be possible to split them up into two value
    #  then adding them to fit sparse mask lines

    success = False
    virtual_state = state.copy()
    offset = 0

    while not success:
        if offset == 1:
            offset = 2
        # Apply the offset (but assure we do not have an offset of size 1 (too small for drawing white space))
        virtual_state.y = state.y + state.dp.y * offset
        no_op_color_code = virtual_state.current_code

        try:
            # Dry run the pixel at the end of the NO_OP
            #virtual_state = state.copy()
            if offset:
                if not 0 <= virtual_state.y < mask.shape[1]:
                    raise OutOfBound(target.value if isinstance(target, PietInstruction) else target.required_size)
                draw_pixel(out_image, mask, virtual_state, virtual_state.x, virtual_state.y, PietColors.code2color[no_op_color_code], mode=Mode.PRIORITIZED_VIRTUALIZED)
                #draw_instruction(out_image, mask, virtual_state, TURN_MARGINS,
                #                 PietInstruction(PIT.NO_OP, max(2, offset), modeler_hints=no_op_color_code),
                #                 mode=Mode.PRIORITARY_VIRTUALIZED)

            # Draw the rest of the drawing
            if isinstance(target, PietInstruction):
                draw_instruction(out_image, mask, virtual_state, TURN_MARGINS, target, *args, mode=Mode((False, False, ignore_mask)), **kwargs)
            else:
                target.draw(out_image, mask, virtual_state, TURN_MARGINS, *args, mode=Mode((False, False, ignore_mask)), **kwargs)
            # Drawing successful
            success = True
            # Draw the offset white pixels
            if offset:
                draw_instruction(out_image, mask, state, TURN_MARGINS, PietInstruction(PIT.NO_OP, offset, modeler_hints=no_op_color_code), mode=Mode.FORCE)
            state.recopy(virtual_state)

        except OutOfMask as oom:
            assert not ignore_mask
            # Apply the recommended offset
            offset += oom.recommended_offset
            # Reset the color
            virtual_state.current_code = state.current_code.rotate()

        except ColorConflict:
            # Add 1 offset at the beginning of each color loop
            if virtual_state.current_code is state.current_code:
                offset += 1
            # Loop throught all the colors
            virtual_state.current_code = virtual_state.current_code.rotate()


def _draw_auto_turn(out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS, target: Union[PietInstruction, InstructionBloc]):
    """Attempt to draw the given instruction and if there is not enough space turn to the next line"""
    try:
        # Attempt to draw the instruction
        _draw_auto_offset(out_image, mask, state, TURN_MARGINS, target)

    except OutOfBound as oob:
        # Add a turn
        _draw_auto_offset(out_image, mask, state, TURN_MARGINS, Turn, minimal_newline_size=oob.expected_space, ignore_mask=True)

        # Re draw the instruction, this time there shouldn't be any problem
        _draw_auto_offset(out_image, mask, state, TURN_MARGINS, target)


def _draw_program(program: BasicMessageDisplay, mask: np.ndarray, initial_color: PietColors, instruction_pixels_cumulative_count):
    """
    Reset and run the zigzag drawing process with the given mask.

    :raise NotEnoughSpace(ratio of instructions drawn): raised by _draw_turn if we hit the bottom of the image.
    """
    state, out_image, TURN_MARGINS = _init(mask, initial_color)

    try:
        for instruction, progress in zip(program.instructions, [0] + instruction_pixels_cumulative_count[:-1]):
            _draw_auto_turn(out_image, mask, state, TURN_MARGINS, instruction)

        # Add the termination pattern
        End.draw(out_image, mask, state, TURN_MARGINS)

    except OutOfBound as oob:
        # Debug
        print(f"Fatal error occure at {state}")
        to_image(out_image, mask, PietColors.WHITE.numpy_color, PietColors.BLACK.numpy_color).show()
        raise

    except NotEnoughSpace as nes:
        # Juste add an insight of how mush space we are missing
        nes.missing_space = instruction_pixels_cumulative_count[-1] - progress
        raise

    # Count remaining unused mask space  TODO: there is probably a faster implementation with numpy functions
    unused_space = 0
    for x, y in itertools.product(range(state.x, len(out_image)), range(len(out_image[0]))):
        if mask[x][y] and out_image[x][y] is None:
            unused_space += 1
    return unused_space, out_image


def _redraw_until_size_is_adjusted(
        program: BasicMessageDisplay,
        raw_mask: PILImage.Image,
        initial_color: PietColors,
        expected_precision=0.01,
        expected_remaining_pixel=10,
        use_alpha_mask=True,
        mask_threshold=128,
        invert_mask=False,
        inner_color=None,
        outer_color=None,
        ):
    """
    This function run the whole drawing process multiple times while adjusting the mask size to best fill space.

    :param expected_remaining_pixel: int, first stop limit. If at the end of the drawing there is less than this
            amout of pixel unused in the mask, we stop
    :param use_alpha_mask: bool, if True use the alpha canal as mask on image with alpha enable.
    :param mask_threshold: int, the threshold above which the pixel is considered in the mask.
    :param invert_mask: bool, invert the mask
    :param expected_precision:
    """
    # First estimation of the mask size
    instruction_pixels_cumulative_count = list(
        itertools.accumulate([instr.value for instr in program.instructions]))
    instruction_required_space = instruction_pixels_cumulative_count[-1]
    raw_mask_available_space = to_mask(raw_mask, use_alpha=use_alpha_mask).sum()

    ratio_to_test = (instruction_required_space / raw_mask_available_space) ** 0.5

    # Init dichotomy varibles
    ratio_min = None
    """Biggest image found that is too small and raise a NotEnoughSpace exception 
    (this should be the mask size returning the biggest NotEnoughSpace.progress attribute)"""
    ratio_max = None
    """The smallest image that is enough large to hold all instruction (this should be the mask size returning 
    the smallest remaining_space value), this store the best valid ratio until a better one is found"""
    remaining_pixels, missing_pixels = 0, 0
    best_out_image, best_mask = None, None

    while (ratio_max is None                                                          # We must loop as long as we don't have at least one valid result
           or ((remaining_pixels is None or remaining_pixels > expected_remaining_pixel) # then stop once there is few remaining pixels
               and (ratio_min is None or ratio_max - ratio_min > ratio_max * expected_precision))    # or if we have shrinked the exploration range a lot, indicating that there may not be a perfect solution
           ):
        new_size = tuple(int(dim * ratio_to_test) for dim in raw_mask.size)
        print(f"Testing a new mask ratio: {ratio_to_test} -> size={new_size}")

        # Generate the mask with the size to test
        mask_to_test = to_mask(raw_mask,
                               use_alpha=use_alpha_mask,
                               threshold=mask_threshold,
                               invert=invert_mask,
                               margins=(0, Turn.OTHER_TURN_SIZE + Turn.offset_security + 5, len(End.PATTERN), Turn.RIGHT_RIGHT_TURN_SIZE + Turn.offset_security + 5),
                               resize_to=new_size,
                               )

        # Test the new size
        try:
            remaning_pixels, out_image = _draw_program(program, mask_to_test, initial_color, instruction_pixels_cumulative_count)
        except NotEnoughSpace as nes:
            print("New drawing too small: missing {} pixels".format(nes.missing_space))
            ratio_min = ratio_to_test
            remaining_pixels, missing_pixels = None, nes.missing_space
        else:
            print("New best drawing found: {} pixel remaining".format(remaning_pixels))
            ratio_max = ratio_to_test
            # Save the drawing as the new best output
            best_out_image, best_mask = out_image, mask_to_test
            missing_pixels = None

        # Define a new size to test
        if ratio_max is None:
            # ratio_to_test /= (1 - missing_pixels / instruction_pixels_cumulative_count[-1])**2   # Exploite returned insight to converge faster but can infinitely lead to size too low
            ratio_to_test *= 2                                                                         # Simpler but slower if the first ratio is far from the optimal value
        elif ratio_min is None:
            ratio_to_test /= (1 - remaning_pixels / instruction_pixels_cumulative_count[-1])**2    # Exploite returned insight  to converge faster,
            # ratio_to_test /= 2                                                                       # Simpler but slower if the first ratio is far from the optimal value
        else:
            # dichotomy
            ratio_to_test = (ratio_max + ratio_min) / 2

    # Fill all None value
    # TODO: implement different filling methods: like:
    #       - filling with a given color (different from piet colors)
    #       - surounding the program with a given color border (different from piet colors) then filing anything else with another color
    #       - Filing with a given image (or the mask) while avoiding color conflict with the program
    #       - Filing with random codels to hide the program

    # Convert into a clean PIL image
    return to_image(best_out_image, best_mask, inner_color, outer_color)


def zigzag_modeler(
    program: BasicMessageDisplay,
    mask: PILImage.Image,
    initial_color=PietColors.RED,
    expected_precision=0.001,
    expected_remaining_pixel=10,
    use_alpha_mask=True,
    mask_threshold=128,
    invert_mask=False,
    outer_color: numpy.ndarray = None,
    inner_color: numpy.ndarray = None,
    )  -> PILImage.Image:
    """
    A simple modeler that draw code by zigzaging line by line on the given mask (doesn't support branching)

    :param program: BasicMessageDisplay, the program to process
    :param mask: PILImage.Image, an image to use as the targeted shape
    :param initial_color:
    :param expected_precision:
    :param expected_remaining_pixel:
    :param use_alpha_mask: bool, if True use the alpha canal as mask on image with alpha enable.
    :param mask_threshold: int, the threshold above which the pixel is considered in the mask.
    :param invert_mask: bool, invert the mask
    :param outer_color: PietColors, the color to use for unused pixels outside the mask
    :param inner_color: PietColors, the color to use for unused pixels inside the mask

    :return: PILImage.Image
    """

    assert all([instruction.instruction_type is not PIT.POINTER for instruction in
                program.instructions]), "The zigzag modeler only handle linear programs, as such you can't use the POINTER instruction"

    return _redraw_until_size_is_adjusted(
        program,
        mask,
        initial_color,
        expected_precision=expected_precision,
        expected_remaining_pixel=expected_remaining_pixel,
        use_alpha_mask=use_alpha_mask,
        mask_threshold=mask_threshold,
        invert_mask=invert_mask,
        outer_color=outer_color,
        inner_color=inner_color,
        )
