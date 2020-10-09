#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Piet-Modeler
# Shaping piet esotheric language programs into an image.
# Copyright (C) 2019  Kasonnara <piet-modeler@kasonnara.fr>
#
# This program is free software under the terms of the GNU General Public License


"""
Command line interface of the project
"""

import argparse

import PIL.Image, PIL.ImageShow

from instruction_generation.static_message import BasicMessageDisplay
from modelers.zigzag.modeler import zigzag_modeler

parser = argparse.ArgumentParser(description='Command line interface of the piet_modeler, an application to generate '
                                             'Piet programs matching the shape of an image')
parser.add_argument('mask', help="Path to the target shape image (currently support jpeg and png)")
parser.add_argument('--invert-mask', help="Invert the mask", action='store_true')
parser.add_argument('--output', '-o', nargs='?', help="Output filename")
#parser.add_argument('method', default='zigzag')

input_group = parser.add_mutually_exclusive_group()

input_group.add_argument('--message', '-m', help='Directly provide the text message to encode into a piet program')
input_group.add_argument('--input_file', '-f', help='Provide the text message to encode into a piet program via a ascii file')

args = parser.parse_args()

ALLOWED_EXT = [".png", ".jpeg", ".jpg"]
if not any(args.mask.lower().endswith(ext) for ext in ALLOWED_EXT):
    # FIXME not robust way to deduce file format
    print("Error: Unsupported mask input format. Supported type: {}".format(", ".join(ALLOWED_EXT)))
    exit(2)

if args.input_file is not None:
    with open(args.input_file, 'r') as in_file:
        args.message = in_file.read()
if args.message is None:
    print("Error: You must provide an input text using either '--message' or '--input_file' option")
    exit(1)
else:
    print("Generating Piet code for message: \"{}{}\"".format(args.message[:30], "..." if len(args.message) > 30 else ''))

img = zigzag_modeler(
    BasicMessageDisplay(args.message),
    PIL.Image.open(args.mask),
    use_alpha_mask=args.mask.lower().endswith(".png"),
    invert_mask=args.invert_mask,
    )

if args.output:
    img.save(args.output)
else:
    PIL.ImageShow.show(img)
