from eye_img import eye_img
from camera import take_picture
import numpy as np
import skimage.io as io

def user_interface():
    while True:
        choice = input("Upload (u) or take a photo (c)? ")
        if choice=="u":
            filepath = input("Filepath: ")
            pic = io.imread(str(filepath))
            break
        elif choice=="c":
            pic = take_picture()
            break
        else:
            print("Invalid input. Try again. ")
        eye_img(pic)