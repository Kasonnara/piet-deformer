# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
Basic instruction generation able to generate very simple code only able to print a fixed message
"""
import functools
from typing import List, Tuple

from core.types import PietInstruction, PietInstructionTypes
from instruction_generation.common import Program


class BasicMessageDisplay(Program):
    def __init__(self, message: str):
        self.message = message.encode('ascii')

        self._instructions = [
            instruction
            for char_value in self.message
            for instruction in self.push_value(char_value) + (PietInstruction(PietInstructionTypes.OUT_CHR, 1),)
            ]

    @staticmethod
    @functools.lru_cache
    def push_value(value: int) -> Tuple[PietInstruction]:
        # basic optimization
        # TODO there is probably more optimal sets of instruction that need less pixels
        d, u = value//10, value%10

        if u:
            result = [PietInstruction(PietInstructionTypes.PUSH, u)]
        else:
            result = []
        if d:
            if d > 1:
                result.append(PietInstruction(PietInstructionTypes.PUSH, d))
                result.append(PietInstruction(PietInstructionTypes.PUSH, 10))
                result.append(PietInstruction(PietInstructionTypes.MULTIPLY, 1))
            else:
                result.append(PietInstruction(PietInstructionTypes.PUSH, 10))
        if d and u:
            result.append(PietInstruction(PietInstructionTypes.ADD, 1))

        return tuple(result)

    @property
    def instructions(self) -> List[PietInstruction]:
        return self._instructions




