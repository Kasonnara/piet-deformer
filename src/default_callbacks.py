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

from piet_common import *

def push_dcb(stack, dp, cc, value, out_log):
    stack.append(value)
    return stack, dp, cc, out_log

def pop_dcb(stack, dp, cc, value, out_log):
    stack.pop()
    return stack, dp, cc, out_log

def pointer_dcb(stack, dp, cc, value, out_log):
    v = stack.pop()
    if v is None:
        raise Exception("Comportement imprévisible sur pointer")
    for k in range(v%4):
        dp = DPdir.next(dp)
    return stack, dp, cc, out_log

def switch_dcb(stack, dp, cc, value, out_log):
    v = stack.pop()
    if v is None:
        raise Exception("Comportement imprévisible sur switch")
    cc = cc * (-1 ** v)
    return stack, dp, cc, out_log

def add_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        stack.append(None)
    else:
        stack.append(v1 + v2)
    return stack, dp, cc, out_log

def substract_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        stack.append(None)
    else:
        stack.append(v2 - v1)
    return stack, dp, cc, out_log

def multiply_op_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        stack.append(None)
    else:
        stack.append(v1 * v2)
    return stack, dp, cc, out_log

def divide_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        stack.append(None)
    else:
        stack.append(v2 // v1)
    return stack, dp, cc, out_log

def mod_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        stack.append(None)
    else:
        stack.append(v2 % v1)
    return stack, dp, cc, out_log

def not_dcb(stack, dp, cc, value, out_log):
    v = stack.pop()
    if v is None:
        stack.append(None)
    else:
        stack.append(1 if v == 0 else 0)
    return stack, dp, cc, out_log

def greater_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        stack.append(None)
    else:
        stack.append(v1 < v2)
    return stack, dp, cc, out_log

def duplicate_dcb(stack, dp, cc, value, out_log):
    stack.append(stack[-1])
    return stack, dp, cc, out_log

def roll_dcb(stack, dp, cc, value, out_log):
    v1 = stack.pop()
    v2 = stack.pop()
    if v1 is None or v2 is None:
        raise Exception("Comportement imprévisible sur roll")
    else:
        rollable = stack[-v2:]
        stack = stack[:-v2] + rollable[-v1:] + rollable[:-v1]
    return stack, dp, cc, out_log

def in_dcb(stack, dp, cc, value, out_log):
    stack.append(None)
    return stack, dp, cc, out_log

def out_int_dcb(stack, dp, cc, value, out_log):
    v = stack.pop()
    if v is None:
        out_log = out_log + "?"
    else:
        out_log = out_log + str(v)
        print(out_log)
    return stack, dp, cc, out_log

def out_chr_dcb(stack, dp, cc, value, out_log):
    v = stack.pop()
    if v is None:
        out_log = out_log + "?"
    else:
        out_log = out_log + chr(v%256) # BUGGGG
        #print(out_log)
    return stack, dp, cc, out_log

def None_dcb(stack, dp, cc, value, out_log):
    return stack, dp, cc, out_log

def Impossible_dcb(stack, dp, cc, value, out_log):
    raise Exception("Opération interdite")

default_callbacks_func = [[Impossible_dcb, push_dcb, pop_dcb],
                          [add_dcb, substract_dcb, multiply_op_dcb],
                          [divide_dcb, mod_dcb, not_dcb],
                          [greater_dcb, pointer_dcb, switch_dcb],
                          [duplicate_dcb, roll_dcb, in_dcb],
                          [in_dcb, out_int_dcb, out_chr_dcb]]
