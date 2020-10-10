# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
<Module docstring>
"""
import pathlib
import subprocess
import tempfile
from typing import Union, Tuple

from PIL import Image as PILImage


def run_npiet(image: Union[PILImage.Image], timeout=60, quiet=True) -> Tuple[str, bool]:
    """
    Run npiet program on the given piet code image

    :param image: A path to an image or directly a PIL image
    :param timeout: Set a timeout for programs that doesn't end.
    :return: the output of npiet, and a boolean indicating if the program has timeout
    """
    if isinstance(image, PILImage.Image):
        tmp_image_file = tempfile.NamedTemporaryFile(prefix='tmp_piet_modeler_tests_', suffix=".png")
        image_path = pathlib.Path(tmp_image_file.name)
        image.save(image_path)
    else:
        tmp_image_file = None
        image_path = pathlib.Path(image)
    assert image_path.exists(), "There is no file: {}".format(image_path)

    try:
        cmd_results = subprocess.run(["npiet", "-q" if quiet else "", str(image_path)],
                                     capture_output=True, timeout=timeout, )
        stdout, has_timeout = cmd_results.stdout.decode(), False
        assert cmd_results.stderr == b"", "npiet returned an error: {}".format(cmd_results.stderr)
    except subprocess.TimeoutExpired as timeout_exception:
        stdout, has_timeout = timeout_exception.stdout.decode(), True

    if tmp_image_file is not None:
        tmp_image_file.close()

    return stdout, has_timeout
