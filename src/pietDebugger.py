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

import sys

import piet_deformer
import piet_common

id = 0
nb_step = 0
direction = {(0,1):"droite", (0,-1):"gauche", (1,0):"bas", (-1,0):"haut"}

def print_status(cmd, stack, dp, cc, value, out_log,x, y, next_pixel, last_dp, last_cc ):
    global id
    print(
    """---------------
    id = {}
    cmd = {}
    stack = {}
    dp = {} to {} 
    cc = {} to {}
    value = {}
    out_log = {}    
    position = ({}, {}) to {}
    """.format(id, piet_common.cmds2str[cmd[0]][cmd[1]], stack, direction[last_dp], direction[dp], last_cc, cc, value, out_log, x ,y, next_pixel))
    id += 1
    global nb_step
    if nb_step>0:
        nb_step -= 1
    else:
        print(">>> nb step to pass ? ")
        nb_step = int(input())

step_by_step_print_callbacks = [[print_status for k in range(3)]for k in range(6)]

if __name__ == '__main__':
    print("Debbuging file :", sys.argv[1])
    reader = piet_deformer.BasicReader(step_by_step_print_callbacks)
    raw_code, out_log = reader.readFile(sys.argv[1])
    print("End")