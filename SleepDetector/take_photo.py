from camera import take_picture
import skimage.io as io
from PIL import Image

pic = take_picture()
im = Image.fromarray(pic)

filename = input("Filename? ")

im.save(f"{filename}.jpeg")
