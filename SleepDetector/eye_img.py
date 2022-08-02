import numpy as np
from facenet_models import FacenetModel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def find_eyes(image_data, *, box_threshold=0.97, prop_const=0.13, disp=False):
    """
    Displays an image with boxes around people's faces and labels them with names.
    Parameters
    ----------
    image_data : numpy.ndarray, shape-(R, C, 3) (RGB is the last dimension)
        Pixel information for the image.
    """
    i = 0
    leftBox = []
    rightBox = []
    
    while i < 4:
        model = FacenetModel()
        
        # in the future, only analyze the highest probability face?
        boxes, probabilities, landmarks = model.detect(image_data)
        face_detected = True

        if boxes is None or probabilities[0]<box_threshold:
            i+=1
            image_data = cv.rotate(image_data, cv.ROTATE_90_CLOCKWISE)
            continue
        
        box = boxes[0]
        prob = probabilities[0]
        
        lefteye = landmarks[0][0]
        righteye = landmarks[0][1]

        # boxes in form [left,top,right,bottom]

        radius = ((box[3] - box[1]) + (box[2] - box[0]))/2 * prop_const

        leftBox = np.array([lefteye[0] - radius, 
                            lefteye[1] - radius, 
                            lefteye[0] + radius,
                            lefteye[1] + radius])
        rightBox = np.array([righteye[0] - radius, 
                            righteye[1] - radius, 
                            righteye[0] + radius,
                            righteye[1] + radius])

        leftBox = np.round(np.array(leftBox)).astype(int)
        rightBox = np.round(np.array(rightBox)).astype(int)
        
        break

    if len(leftBox)>0 and disp:
        fig, ax = plt.subplots()
        ax.imshow(image_data)

        ax.add_patch(Rectangle(leftBox[:2], *(leftBox[2:] - leftBox[:2]), fill=None, lw=2, color="yellow"))
        ax.add_patch(Rectangle(rightBox[:2], *(rightBox[2:] - rightBox[:2]), fill=None, lw=2, color="yellow"))
        
        ax.plot(lefteye[0], lefteye[1], "+", color="blue")
        ax.plot(righteye[0], righteye[1], "+", color="blue")
    
    return leftBox, rightBox, image_data

def eye_img(pic, *, disp=False):
    if pic.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        pic = pic[..., :-1]  # png -> RGB

    left, right, pic = find_eyes(pic)

    if len(left)==0:
        print("FACE NOT FOUND")
        return None

    
    gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
    
    # boxes in form [left,top,right,bottom]
    

    lefteye = gray[left[1]:left[3], left[0]:left[2]]
    righteye = gray[right[1]:right[3], right[0]:right[2]]
    
    lefteye = cv2.resize(lefteye, (24,24), interpolation = cv2.INTER_LINEAR)
    righteye = cv2.resize(righteye, (24,24), interpolation = cv2.INTER_LINEAR)
    
    if disp:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(lefteye, cmap="gray")
        axes[1].imshow(righteye, cmap="gray")
    
    return lefteye, righteye
