# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License

import unittest
from pathlib import Path

import PIL.Image as PILImage

# TODO make a test fixture that assert npiet is installed
from instruction_generation.static_message import BasicMessageDisplay
from modelers.zigzag.modeler import zigzag_modeler
from tests.utils import run_npiet


class ZigZagProducedCodeTestCase(unittest.TestCase):

    def test_produced_piet_code_print_input_message(self, test_message="Hello world!"):
        # Generate piet code
        program = BasicMessageDisplay(test_message)
        mask = PILImage.open(Path("sample") / "masks" / "circle.png")
        piet_code_image = zigzag_modeler(program, mask)

        # Run it with npiet interpreter
        piet_program_output, has_timeout = run_npiet(piet_code_image)

        # Check
        self.assertFalse(has_timeout)
        self.assertEqual(test_message, piet_program_output)


if __name__ == '__main__':
    unittest.main()
