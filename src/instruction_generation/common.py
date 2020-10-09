# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
Common abstract class
"""
from abc import abstractmethod, ABC
from typing import List

from core.types import PietInstruction


class Program(ABC):
    @property
    @abstractmethod
    def instructions(self) -> List[PietInstruction]:
        pass

    def estimated_pixel_lenght(self) -> int:
        return sum([pied_instr.value for pied_instr in self.instructions])
