import matplotlib.image as Image
import sys

img = Image.imread(sys.argv[1])
print("image =",img)
print("type =",img.dtype)