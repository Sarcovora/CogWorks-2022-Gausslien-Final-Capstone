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
    return eye_img(pic,disp=True)

if (not torch.cuda.is_available()):
    print("USING CPU")
    device = torch.device('cpu')
else:
    print("USING GPU")
    device = torch.device('cuda')

model = torch.load("model.pb", map_location=device)

left, right = user_interface()

left = torch.tensor(left, device=device).reshape(1,1,24,24).float()/255
right = torch.tensor(right, device=device).reshape(1,1,24,24).float()/255

left = torch.argmax(model(left)).item()
right = torch.argmax(model(right)).item()

if left or right:
    print("Eyes open")
else:
    print("Eyes closed")