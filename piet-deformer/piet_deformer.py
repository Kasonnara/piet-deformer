import matplotlib.image as Image
import numpy as np

from default_callbacks import *
from piet_modeler import *
from piet_common import *


white = "#FFFFFF"
black = "#000000"
all_colors = [white, black] + list(colors2code.keys())

def get_pixel_color(pixel):
    color = "#"
    byfloat = False
    if (pixel<=1).all():
        # adapter si on a un jpg [0-255] ou un png [0.-1.]
        # TODO a améliorer
        byfloat = True
    for k in range(3):
        h= None
        if byfloat:
            h = str(hex(int(pixel[k]*255)))[2:4]
        else:
            h = str(hex(pixel[k]))[2:4]
        color = color + (h if len(h)==2 else "0"+h).upper()
    if color not in all_colors:
        raise ValueError("Color not valid {} -> {}".format(pixel, color))
    return color


class Codel:
    def __init__(self, x, y, color, codelMap):
        self.pixel_list = [(x,y)]
        self.color = color
        codelMap[x][y] = self
        self.cmd = None

    def merge(self, other_codel, codelMap, all_codels):
        if not self.color == other_codel.color:
            raise Exception("Merging two groups of different color")
        self.pixel_list = self.pixel_list + other_codel.pixel_list
        for p in other_codel.pixel_list:
            codelMap[p[0]][p[1]] = self
        all_codels.remove(other_codel)

    def add_pixel(self, x, y, codelMap):
        self.pixel_list.append((x,y))
        codelMap[x][y] = self

    @staticmethod
    def get_all_codels(img):
        dim = img.shape
        codelMap = [[None for j in range(dim[1])] for i in range(dim[0])]
        all_codels = []

        for i in range(dim[0]):
            for j in range(dim[1]):
                color = get_pixel_color(img[i, j])
                same_group_top = (i > 0) and (codelMap[i-1][j].color == color)
                same_group_left = (j > 0) and (codelMap[i][j-1].color == color) and not(codelMap[i-1][j] is codelMap[i][j-1])
                if same_group_left and same_group_top:
                    # merge groups
                    codelMap[i - 1][j].add_pixel(i, j, codelMap)
                    codelMap[i-1][j].merge(codelMap[i][j-1], codelMap, all_codels)
                else:
                    if same_group_left:
                        codelMap[i][j-1].add_pixel(i, j, codelMap)
                    elif same_group_top:
                        codelMap[i-1][j].add_pixel(i, j, codelMap)
                    else:
                        all_codels.append(Codel(i, j, color, codelMap))
            print("Detecting blocks : {}%".format(100*i//dim[0]))
        return all_codels, codelMap

    def get_next_pixel(self, dp, cc, codelMap, try_counter=0):
        if try_counter == 8:
            return None

        img_dim = len(codelMap), len(codelMap[0])
        def order_key(pixel):
            return dp[0]*pixel[0] + dp[1]*pixel[1], cc*dp[1]*pixel[0] + cc*dp[0]*pixel[1]

        out_pixel = max(self.pixel_list, key=order_key)
        next_pixel = out_pixel

        next_color = white
        white_flag = False
        while next_color == white:
            # Compute next pixel coordonnate
            next_pixel = (next_pixel[0]+dp[0], next_pixel[1]+dp[1])
            # Check if in bound
            if 0 <= next_pixel[0] < img_dim[0] and 0 <= next_pixel[1] < img_dim[1]:
                # get pixel color
                next_color = codelMap[next_pixel[0]][next_pixel[1]].color
                if next_color == white:
                    white_flag = True
            else:
                # Rotate
                return self.get_next_pixel(dp if try_counter % 2 == 1 else DPdir.next(dp), -cc, codelMap,
                                           try_counter=try_counter + 1)
        if next_color == black:
            # Rotate
            return self.get_next_pixel(dp if try_counter % 2 == 1 else DPdir.next(dp), -cc, codelMap,
                                       try_counter=try_counter + 1)
        else:
            return next_pixel, dp, cc, white_flag

    def __sub__(self, other):
        c1 = colors2code[self.color]
        c2 = colors2code[other.color]
        return (c1[0]-c2[0])%6, (c1[1]-c2[1])%3

    value = property(lambda self: len(self.pixel_list))

class BasicReader:
    """Lit l'image et recompose le code en éliminant les contraintes géographique du piet (déplacment, direction, etc.)
    Ne fonctionne que pour les programmes simple sans aiguillage car il n'est pas capablede gérer les structure de controle
    """
    def __init__(self, callbacks):
        """:parameter callbacks liste de liste de fonctions de dimension 6x3 pour chaque commande du langage"""
        self.callbacks = callbacks

    def readFile(self, filename, insert_input=False):
        img = Image.imread(filename)
        print("Preprocessing all blocks")
        all_codels, codelMap = Codel.get_all_codels(img)

        print("List of blocks:")
        for codel in all_codels[:5]:
            print("block color =", codel.color, "pixel_list =", codel.pixel_list)

        dp, cc = DPdir.droite, CCdir.gauche
        x, y = (0, 0)
        stack = []
        out_log = ""

        raw_code = []

        stop = False
        i = 0
        ratio = len(all_codels)/100
        print("Reading code")
        while not stop:
            #print("truc",i, dp, cc, x, y, stack)
            last_dp, last_cc = dp, cc
            result = codelMap[x][y].get_next_pixel(dp, cc, codelMap)
            if result is None:
                stop = True
            else:
                next_pixel, dp, cc, white_flag = result
                if not white_flag:
                    # Register command
                    next_codel = codelMap[next_pixel[0]][next_pixel[1]]
                    current_codel = codelMap[x][y]
                    next_cmd = next_codel - current_codel
                    #print("next_cmd:",cmds[next_cmd[0]][next_cmd[1]], "value:", current_codel.value)
                    raw_code.append((next_cmd, current_codel, next_codel))
                    if next_codel.cmd is None:
                        current_codel.cmd = next_cmd
                    else:
                        raise Exception("Loop detected, can't be handled. (at position {} from pixel {}, dp = {}, "
                                        "cc = {}, next_cmd = {})".format(next_pixel, (x,y), dp, cc, cmds2str[next_cmd[0]][next_cmd[
                            1]]))
                    value = current_codel.value
                    stack, dp, cc, out_log = default_callbacks_func[next_cmd[0]][next_cmd[1]](stack, dp, cc, value, out_log)
                    if self.callbacks[next_cmd[0]][next_cmd[1]] is not None:
                        self.callbacks[next_cmd[0]][next_cmd[1]](next_cmd, stack, dp, cc, value, out_log, x,y, next_pixel, last_dp, last_cc )
                    # ================ spécial code =========
                    # edite le code pour ajouter un input quand [press enter] est tapé
                    if insert_input:
                        if next_cmd == (5, 2) and out_log[-13:] == "[press enter]":
                            print("Inserting pause")
                            raw_code.append(((5,0), None, None))
                            raw_code.append(((0,2), None, None))
                    # =======================================
                x,y = next_pixel

                i += 1
                if i % ratio == 0:
                    print("Analyzing : {}%".format(10 * i // ratio))

        return raw_code, out_log

    def print_code(self, raw_code):
        for i,instruction in enumerate(raw_code):
            print("{}){}\tvalue={}".format(i,
                                           cmds2str[instruction[0][0]][instruction[0][1]],
                                           1 if instruction[1] is None else instruction[1].value))

def lineariser_basic(raw_code):
    """Linearise le code piet généré automatiquement par le convertisseur foogol->piet
    Ce convertisseur se contente de faire des allez retour, donc simple a linéariser
    il est trop basique pour d'autres codes piet qui utilisent plus le DP et le CC"""
    k = 2
    to_keep = [True for k in range(len(raw_code))]
    while k < len(raw_code):
        if raw_code[k][0] == (3,1): # Pointer cmd
            # check the entire pattern
            #print("prev",raw_code[k-1][0], "anteprev",raw_code[k-2][0], "next",raw_code[k+1][0], "anteprev value", raw_code[k-2][1].value)
            if (raw_code[k+1][0] == (3,1) # 2nd pointer
                and raw_code[k-1][0] == (4,0) # previous duplicate
                and raw_code[k-2][0] == (0,1) # previous previous push
                and (raw_code[k-2][1].value == 1 or raw_code[k-2][1].value == 3) # rotation value
            ):
                to_keep[k-2:k+2] = [False]*4
                k += 1
            else:
                raise Exception("Code not linearisable, patern doesn't match at cmd "+str(k))
        k += 1
    return [(cmd[0], 1 if cmd[1] is None else cmd[1].value) for k,cmd in enumerate(raw_code) if to_keep[k]]


if __name__ == '__main__':
    reader = BasicReader([[None for k in range(3)]for l in range(6)])
    raw_code , out_log = reader.readFile("../message-secret/message.piet", insert_input=True)
    reader.print_code(raw_code)
    print("Out :", out_log)

    print("Linearising code ...")
    linearized_code = lineariser_basic(raw_code)
    #generate_line_code(linearized_code)
    model = Image.imread("../message-secret/red-heart-hi-2.png")[:,:,1]<0.5
    out_img = map_code(model, linearized_code)
    print(out_img)
    print("type =", out_img.dtype)
    Image.imsave("../message_modele.png", out_img)
    print("Model sauvé")

