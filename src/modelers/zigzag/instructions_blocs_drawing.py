# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License

"""
Blocs of instructions that can be drawn sequentially by the zigzag modeler
"""

import itertools
from typing import Optional, List

import numpy

from core.types import PietInstructionTypes as PIT, PietInstruction as PI, PietColors as C, DPDir, \
    State, PietInstruction, PietColors
from core.exceptions import NotEnoughSpace
from core.types import Mode
from modelers.zigzag.instruction_drawing import draw_instruction


class InstructionBloc(type):
    def __call__(cls, *args, **kwargs):
        raise AssertionError("InstructionBloc classes are singletons and must not be instanced. Use the class itself instead.")

    def draw_instructions(cls, out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS,
             bloc_instructions: List[PietInstruction], *args, mode=Mode.NORMAL, **kwargs):
        """
            Draw the given instruction, and update draw state (state.x, state.y, state.current_code and state.dp).

            :param out_image: List[List[Optional[PietColors]]], the image to draw on.
            :param mask: numpy.ndarray[bool], the mask whose shape we are trying to match.
            :param state:  State, the current state of the Piet Interpreter when entering this bloc.

            :raise OutOfBound, OutOfMask, ColorConflict: see check_instruction_drawable.
            """
        if mode is Mode.NORMAL:
            virtual_state = state.copy()
            cls.draw_instructions(out_image, mask, virtual_state, TURN_MARGINS, bloc_instructions, mode=Mode.VIRTUALIZED_DRY)
            mode = Mode.FORCE

        for instruction in bloc_instructions:
            draw_instruction(out_image, mask, state, TURN_MARGINS, instruction, mode=mode)


class Turn(metaclass=InstructionBloc):
    INSTRUCTIONS = {
        # Double Right turn
        (DPDir.RIGHT, DPDir.LEFT): [PI(PIT.PUSH, 1), PI(PIT.DUPLICATE), PI(PIT.POINTER, modeler_hints=1),
                                    PI(PIT.POINTER, modeler_hints=1)],
        # Double Left turn
        (DPDir.LEFT, DPDir.RIGHT): [PI(PIT.PUSH, 3), PI(PIT.DUPLICATE, 1), PI(PIT.POINTER, modeler_hints=3),
                                    PI(PIT.POINTER, modeler_hints=3)],
        # Right Left turn
        (DPDir.RIGHT, DPDir.RIGHT): [PI(PIT.PUSH, 1), PI(PIT.PUSH, 3), PI(PIT.POINTER, modeler_hints=1),
                                     PI(PIT.POINTER, modeler_hints=3)],
        # Left Right turn
        (DPDir.LEFT, DPDir.LEFT): [PI(PIT.PUSH, 3), PI(PIT.PUSH, 1), PI(PIT.POINTER, modeler_hints=3),
                                   PI(PIT.POINTER, modeler_hints=1)],
        }


    RIGHT_RIGHT_TURN_SIZE = 3  # = sum(size for _, size in TURN_INITIALIZATION[( 1, -1)]
    OTHER_TURN_SIZE = 5  # = sum(size for _, size in TURN_INITIALIZATION[ ... ]
    offset_security = 3

    @staticmethod
    def count_available_space(mask_line: List[bool], required_consecutive_space=0) -> int:
        """
        Count space available on the given partial line.
        (But start counting only after founding an area of at least <required_consecutive_space> consecutive cells)

        :param mask_line: the mask partial line indicating which cell are available
        :param required_consecutive_space: int, the minimal number of space required for the next piet instruction
        :return: int
        """
        consecutive_count = 0
        extra_space = 0
        for is_cell_available in mask_line:
            # First assure we have the required consecutive minimal size
            if consecutive_count < required_consecutive_space:
                if is_cell_available:
                    consecutive_count += 1
                else:
                    # Mask is not consecutive, we cannot use this space
                    # TODO: here there is a potential space lose, especially if minimal_newline_size is high and/or
                    #  the mask is sparse. In a more advanced version of the modeler, I may be a good idea may to
                    #  implements some optional strategies that could decide to ignore little gaps in the mask; or
                    #  to fill these little mask island with dummy codels in order to avoid to many blank space,
                    #  skipped lines and in the end getting
                    consecutive_count = 0
            # Then in a second time count how many extra space we have after that in case both side meet the first requirement
            else:
                if is_cell_available:
                    extra_space += 1
        if consecutive_count >= required_consecutive_space:
            return consecutive_count + extra_space
        else:
            return 0
        """ 
        # Old implementation (faster by exploiting numpy functions, but doesn't handle minimal_newline_size)
        return = sum(mask_line)
        """


    @classmethod
    def find_new_line(cls, out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS, minimal_newline_size: int=1):
        """
        Decide how many line to skip and on which side we will turn.
        the first line with enough space and the best side to turn on this line

        :param minimal_newline_size: int, the required minimum number of available cells on the new line

        :return: the number of line to go down, and the new DPDir to turn to
        """
        num_skipped_line = 0
        free_line_found = False
        new_dp: Optional[DPDir] = None
        while not free_line_found:
            num_skipped_line += 1
            if 1 < num_skipped_line < 4:
                # Cannot skip exactly 2 or 3 line because we need a gap for placing he white space
                # TODO implement a special turn pattern with a 2 or 3 pixel Pointer instruction for that
                num_skipped_line = 4

            if state.x + num_skipped_line >= mask.shape[0]:
                # Check that we don't hit the bottom of the image
                raise NotEnoughSpace()

            # Count space available on each size of the line:
            #     - the fact that a double right turn is smaller than other turns
            #     - the next command require a consecutive mask space of size minimal_newline_size
            #     - turn margins
            # TODO instead of choosing left or right it may be a better stategy to go directly at the farest pixel and then turn to use the entire line, the only issue is that the turn pixels will often be out of mask
            left_space = cls.count_available_space(
                # FIXME This formula isn't really precise: offset due to the turn can be lower (if not NO_OP is required) of larger if a big offset is needed! the "+2" is here as a margin in case of a short NO_OP but it may not be sufficient and on bad luck may end up selecting a line without enought space
                mask[state.x + num_skipped_line][max(0, state.y + (Turn.RIGHT_RIGHT_TURN_SIZE - 1 if state.dp.y > 0 else -(Turn.OTHER_TURN_SIZE + 1 + 2))):
                                                 TURN_MARGINS[0]: -1],
                minimal_newline_size,
                )
            right_space = cls.count_available_space(
                mask[state.x + num_skipped_line][max(0, state.y + (Turn.OTHER_TURN_SIZE + 1 + 2 if state.dp.y else -Turn.OTHER_TURN_SIZE + 1)): TURN_MARGINS[1]: 1],
                minimal_newline_size,
                )

            # Choose the side with the more space
            if left_space or right_space:
                free_line_found = True
                new_dp = DPDir.LEFT if left_space > right_space else DPDir.RIGHT
        return num_skipped_line, new_dp

    @classmethod
    def draw(cls, out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS, minimal_newline_size: int,
             *args, **kwargs):
        """
        Draw a line flip pattern, and update draw state

        :raise ColorConflict: If there is a color conflict with previous line that must be solve with an offset
        :raise NotEnoughSpace: If we hit the bottom of the image
        """
        # TODO: improve the turn process to best match the mask. For example, by turning earlier (before exiting
        #  the mask) if an option <strict> is set to True; or by turning later by extending the pointer codel size if we
        #  are not at the end of the mask.

        # TODO: It may be possible to design other turn system using black pixels. This could avoid using PUSH and
        #  POINTER instruction, and even turn in the middle of a big push instruction

        # FIXME: There will probably be a issue with a big minimal_newline_size value (e.g. push of a big value)
        #  on a sparse mask without sufficient space to hold the instruction. Resulting is skipping many lines. But it's
        #  clear that this modeler with its simple behaviour will behave badly on spars masks in many ways.
        assert state.dp.x == 0

        num_skipped_line, new_dp = cls.find_new_line(out_image, mask, state, TURN_MARGINS, minimal_newline_size)
        assert num_skipped_line != 2, "NotImplementedError"

        instructions = (
            Turn.INSTRUCTIONS[state.dp, new_dp][:3]
            # Insert optional white space when skipping multiple lines
            # (TODO: in a more advanced version, we can chose color here)
            + ([PI(PIT.NO_OP, num_skipped_line)] if num_skipped_line > 1 else [])
            + Turn.INSTRUCTIONS[state.dp, new_dp][3:]
            )

        # Dry run for color conflict (only check the pixels drawn on the current line, the pixels drawn on the lower
        # lines directly under the turn pattern pixels will never conflict)
        cls.draw_instructions(out_image, mask, state.copy(), TURN_MARGINS, instructions[:3], mode=Mode.PRIORITIZED_VIRTUALIZED)

        # Draw the turn
        cls.draw_instructions(out_image, mask, state, TURN_MARGINS, instructions, mode=Mode.FORCE)


class End(metaclass=InstructionBloc):
    PATTERN = [[None,    None,    C.BLACK, None   ],
               [None,    C.WHITE, C.RED,   C.BLACK],
               [C.BLACK, C.RED,   C.RED,   C.BLACK],
               [None,    C.BLACK, C.BLACK, None   ]]

    @classmethod
    def draw(cls, out_image: List[List[Optional[PietColors]]], mask: numpy.ndarray, state: State, TURN_MARGINS, end_type='downward',
             *args, **kwargs):
        if end_type == 'downward':
            assert (state.x + len(cls.PATTERN) < mask.shape[0]
                    and 0 <= state.y - 1 and state.y + 2 < mask.shape[1]), "Termination pattern will hit boudaries, this shouldn't happen, you may have not respected margins"

            initial_dp_y = state.dp.y

            # Rotate downwards
            instructions = [PI(PIT.PUSH, 1), PI(PIT.POINTER, modeler_hints=1)] if state.dp.y > 0 else [PI(PIT.PUSH, 3), PI(PIT.POINTER, modeler_hints=3)]
            cls.draw_instructions(out_image, mask, state, TURN_MARGINS, instructions, mode=Mode.FORCE)  # FIXME use safe draw

            # Draw the end pattern itself
            for i, j in itertools.product(range(len(cls.PATTERN)), range(len(cls.PATTERN[0]))):
                assert state.x + i < len(out_image), "This check line shouldn't be necessary due to margins"
                if cls.PATTERN[i][j] is not None:
                    assert out_image[state.x + i][state.y + (j - 1) * initial_dp_y] is None, \
                        "Drawing the end pattern shouldn't override already drawn pixels"
                    out_image[state.x + i][state.y + (j - 1) * initial_dp_y] = cls.PATTERN[i][j]
        else:
            raise NotImplementedError()
