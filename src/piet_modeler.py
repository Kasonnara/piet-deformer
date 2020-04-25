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

import matplotlib.image as Image
import matplotlib.pyplot as plt
import numpy as np

from piet_common import *


# ================== LINÉARISER =======================
def generate_line_code(linear_code):
    length = sum([value for cmd,value in linear_code])+1
    out_img = np.zeros((1,length,3))
    index = 0
    previous_code = (0,1)
    for cmd,value in linear_code:
        next_code = add_codes(previous_code, cmd)
        for k in range(value):
            out_img[0,index + k] = code2color[previous_code]
        previous_code = next_code
        index += value
    out_img[0,-1] = code2color[previous_code]
    print(out_img)
    Image.imsave("linarized_code.png", out_img)
# ======================================================


def extract_model(filename):
    img = Image.imread(filename)
    alpha = img[:,:,3]
    return alpha > 0.5


def eval_model_size_ratio(model, linearized_code):
    model_size = sum(sum(model))
    print("Taille utilisable du modele =",model_size)
    code_size = sum([value for cmd,value in linearized_code])
    print("Taille aproximative requise du code =", code_size)
    return np.sqrt(code_size*over_fit_ratio/model_size)


def put_line_cmd(out_img, x, y, sens, current_code, cmd_code, cmd_value, simulate=False, restrict2model=True):
    """Insert a block in out_img at the givent position following the same line (to the right on even lines and to
    the left on odd lines) and corresponding to the given cmd_code and cmd_value
    return the new position and the next_required color
    raise ConflictException if it's blocs make conflicts with previous blocs
    """
    if not simulate:
        put_line_cmd(out_img, x, y, sens, current_code, cmd_code, cmd_value, simulate=True,
                     restrict2model=restrict2model)

    current_color = code2color[current_code]
    next_code = add_codes(current_code, cmd_code)
    for k in range(cmd_value):
        #print("sens =", sens, "y+k*sens =", y + k * sens)
        if (y+k*sens >= len(out_img[0]) - (3 if (restrict2model and sens==1) else 0)) or (y+k*sens <= (5 if (restrict2model and sens ==-1) else -1)):
            print("Erreur bordure", "gauche" if (y+k*sens <= (5 if restrict2model and sens ==-1 else 0)) else "droite", "atteinte")
            raise BorderException()
        if x>0 and ((out_img[x - 1][y + k*sens] == current_color).all()):
            print("Erreur conflit")
            raise ConflictException()
        if not(0 < y+(k+1)*sens < len(out_img[0])-1):
            print("Erreur not enought place to put the next pixel")
            raise BorderException()
        if x>0 and (out_img[x-1][y+(k+1)*sens] == code2color[next_code]).all():
            print("Next pixel conflict")
            raise ConflictException()
        if restrict2model and ((out_img[x][y + k*sens] == out_model_color).all()):
            print("Erreur model restrict2model=", restrict2model)
            raise ModelException()
        if not simulate: # Do not draw on simulation mode
            out_img[x][y+k*sens] = current_color
    return x, y+cmd_value*sens, next_code


def put_line_flip(out_img, x, y, sens, current_code, simulate=False):
    """Insert a flip patern
    return the new position and the next_required color
    raise an error if it's blocs make conflicts with previous blocs"""
    # Search best line and side to flip
    i = 0
    searching = True
    new_sens = None
    while searching:
        i += 1
        if i==2:
            i=3
        if x + i >= len(out_img):
            plt.imshow(out_img)
            plt.show()
            raise Exception("End of model, code not finished")
        left_side = sum([out_img[x+i][j][0] > 0.5 for j in range(5,y)])
        right_side = sum([out_img[x + i][j][0] > 0.5 for j in range(y, len(out_img[0]-3))])
        if left_side > 6 or right_side > 5:
            new_sens  = -1 if left_side > right_side else 1
            searching = False
            print("better sens found after",i,"lines :", new_sens)

    if not simulate:
        # Check for conflict
        put_line_flip(out_img, x, y, sens, current_code, simulate=True)


    #xt, yt, current_code_t = put_line_cmd(out_img, x, y, sens, current_code, str2cmd["push"], 1 if sens == 1 else 3, simulate=True, restrict2model=False)
    #if sens == new_sens:
    #    xt, yt, current_code_t = put_line_cmd(out_img, x, y, sens, current_code, str2cmd["push"], 1 if new_sens == -1 else 3,
    #                                          simulate=True, restrict2model=False)
    #else:
    #    xt, yt, current_code_t = put_line_cmd(out_img, xt, yt, sens, current_code_t, str2cmd["duplicate"], 1, simulate=True, restrict2model=False)
    #_, _, current_code_t = put_line_cmd(out_img, xt, yt, sens, current_code_t, str2cmd["pointer"], 1, simulate=True, restrict2model=False)
    #put_line_cmd(out_img, xt + 1, yt+i, new_sens, current_code_t, str2cmd["pointer"], 1, simulate=True, restrict2model=False)

    try:
        # Draw cmd
        #  Push first rotation value
        x, y, current_code = put_line_cmd(out_img, x, y, sens, current_code, str2cmd["push"], 1 if sens == 1 else 3, restrict2model=False)
        # Push second rotation value
        if sens == new_sens:
            # Repush a new value
            x, y, current_code = put_line_cmd(out_img, x, y, sens, current_code, str2cmd["push"],
                                                  1 if new_sens == -1 else 3, restrict2model=False)
        else:
            # Just duplicate previous value
            x, y, current_code = put_line_cmd(out_img, x, y, sens, current_code, str2cmd["duplicate"], 1, restrict2model=False)
        #  Rotate once
        x, y, current_code = put_line_cmd(out_img, x, y, sens, current_code, str2cmd["pointer"], 1, restrict2model=False)
        reinit_code = current_code # en cas de saut de ligne on mémorise l'avant dernière couleur
        #                            pour la repositionner correctement à la fin du saut
        # Rotate twice
        _, _, current_code = put_line_cmd(out_img, x, y, new_sens, current_code, str2cmd["pointer"], 1,
                                          restrict2model=False)
        # offset if needed
        if not simulate:
            for k in range(1, i):
                if k == i - 1:
                    out_img[x+k][y] = code2color[reinit_code]
                else:
                    out_img[x+k][y] = color_white
        x,y = x+i,y
    except BorderException or ModelException or ConflictException:
        if simulate:
            raise
        else:
            raise Exception("Simuation doesn't predict an error!")
    return x, y, new_sens, current_code

def put_end(out_img, x, y, sens, simulate=False):
    if not simulate:
        put_end(out_img, x, y, sens, simulate=True)

    #TODO can be rewrite using numpy opération only!
    for i, line in enumerate(end_patern):
        for j, color in enumerate(line):
            if color is not None:
                rx, ry = x+(i-1), y+(j)*sens + (len(end_patern[0]) if sens == -1 else 0)
                if not(0 <= rx <= len(out_img) and 0 <= ry < len(out_img[0])):
                    print("not enought space for placing the ending")
                    raise BorderException()
                if not((out_img[rx][ry] == model_color).all() or (out_img[rx][ry] == out_model_color).all()):
                    print("End overlaps other pixel")
                    raise ConflictException()
                if not simulate:
                    out_img[rx][ry] = end_patern[i][j]

def add_end(out_img, x, y, sens):
    success = False
    k = 0
    while not success:
        try:
            put_end(out_img, x, y+k*sens, sens)
        except ConflictException:
            k += 1
            print("k =", k, "sens =", sens)
        except BorderException:
            raise Exception("Fatal error can't place ending")
        else:
            success = True


"""
def next_bloc(dim, x, y):
    y += 1
    if y >= dim[1]:
        y = 0
        return None
        if x >= dim[0]:
            raise Exception("image full")
    return x,y


def find_next_pos(out_img, x,y):
    dim = out_img.shape
    x,y = next_bloc(dim)
    white_flag = False
    while out_img[x,y,3] == 0:
        white_flag =  True
        x, y = next_bloc()
    return x,y, white_flag
"""

def add_cmd_block(out_img, x, y, sens, current_code, cmd_code, cmd_value):
    print("inserting new cmd", cmd_code, "value", cmd_value, "with prev color", current_code, "position (",x,",",y,"), sens",sens)
    # check si suffisament de pixel dans le model
    success = False
    k = 0
    code_rotate = current_code
    next_x, next_y, next_code = None, None, None
    while not success:
        try :
            try:
                # ajouter le block
                next_x, next_y, next_code = put_line_cmd(out_img, x, y+k*sens, sens, code_rotate, cmd_code, cmd_value)
            # sinon
            except ConflictException:
                # Essayer de changer la couleur
                #  Si pas de blanc on en ajoute 1
                if k ==0:
                    k=2
                code_rotate = (code_rotate[0]+1)%6, (code_rotate[1] + (1 if code_rotate[0] == 5 else 0))%3
                #  Si toutes les couleur ont été essayées, essayer d'ajouter plus loin
                if code_rotate == current_code:
                    raise ModelException
                else:
                    print("retrying color", code_rotate)
            else:
                success = True
                # compéter le trou
                for i in range(k):
                    out_img[x,y + i*sens,:] = (code2color[current_code] if i == 0 else color_white)
        except ModelException:
            # essayer d'ajouter plus loins
            if k == 0:
                k=2
            else:
                k += 1
            print("retrying decalage", k)
            code_rotate = current_code
        except BorderException:
            print("Fail to place block, adding return")


            # Ajouter un retour la ligne
            k = 0
            flip_success = False
            code_rotate = current_code
            next_x, next_y, next_sens, next_code = None, None, None, None
            while not flip_success:
                try:
                    # essayer un ajout
                    print("try flipping at (",x,",",y+k*sens,")")
                    next_x, next_y, next_sens, next_code = put_line_flip(out_img, x, y+k*sens, sens, code_rotate)
                except ConflictException:
                    # Sinon essayer de changer la couleur jusqu'a ce que ça passe
                    k = 2
                    code_rotate = (code_rotate[0]+1)%6, (code_rotate[1] + (1 if code_rotate[0] == 5 else 0))%3
                    if code_rotate == current_code:
                        raise Exception("Fail critique lors du placement retour")
                    else:
                        print("flip retrying color", code_rotate)
                else:
                    flip_success = True
                    # completer le saut si besoin
                    for i in range(k):
                       out_img[x,y+i*sens,:] = (code2color[current_code] if i == 0 else color_white)
                    k = 0
                    x = next_x
                    y = next_y
                    sens = next_sens
                    current_code = next_code
                    code_rotate = current_code
                    print("flip added successfully")
                    #plt.imshow(out_img)
                    #plt.show()
    return next_x, next_y, sens, next_code

def is_model(i,j,ratio, model):
    j_model = int((j-10)/ratio)
    i_model = int(i / ratio)
    return j_model >= 0 and j_model < len(model[0]) and i_model<len(model) and model[i_model][j_model]

def map_code(model, linearized_code):

    ratio = eval_model_size_ratio(model, linearized_code)
    dim = model.shape
    dim = int(round(dim[0]*ratio)), int(round(dim[1]*ratio))
    out_img = np.array([[
                (model_color if is_model(i,j,ratio, model) else out_model_color)
         for j in range(dim[1]+20)] for i in range(dim[0]+4)], np.uint8)
    plt.imshow(out_img)
    # search first usable pixel of the model
    x, y = 0, 0
    sens = 1
    current_code = (0,1)
    #out_img[0][0] = code2color[current_code]
    i = 0
    for cmd_code, cmd_value in linearized_code:
        i +=1
        print("cmd",i, "/", len(linearized_code))
        x,y, sens, current_code = add_cmd_block(out_img, x, y, sens, current_code, cmd_code, cmd_value)
    print("Adding end patern")
    add_end(out_img, x, y, sens)

    plt.imshow(out_img)

    print("cleanning the image...")
    i = 0
    while i<len(out_img):
        j = 0
        while j< len(out_img[i]):
            if (out_img[i,j] == model_color).all():
                out_img[i, j] = color_white if i <= x else code2color[0][0]
            if (out_img[i,j] == out_model_color).all():
                out_img[i,j] = color_white
            j+=1
        i+=1
    plt.show()
    return np.array(out_img)


