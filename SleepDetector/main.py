from camera import take_picture
import numpy as np
import skimage.io as io
import torch

from eye_img import eye_img
from eyemodel import EyeModel

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

if (not torch.cuda.is_available()):
    print("NOT USING GPU")
    model = torch.load("model.pb", map_location=torch.device('cpu'))
else:
    print("USING GPU")
    model = torch.load("model.pb", map_location=torch.device('cuda'))

left, right = user_interface()

left = torch.tensor(left).reshape(1,1,24,24).float()/255
right = torch.tensor(right).reshape(1,1,24,24).float()/255

left = torch.argmax(model(left)).item()
right = torch.argmax(model(right)).item()

def printer(eyeopen):
    return "open" if eyeopen else "closed"

print("Left eye "+printer(left))
print("Right eye "+printer(right))