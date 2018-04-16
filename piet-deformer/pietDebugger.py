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