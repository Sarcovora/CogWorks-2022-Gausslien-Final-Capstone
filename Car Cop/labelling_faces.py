{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2372a261",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e130ca8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from PIL import Image\n",
    "from Car Cop import query_database, cos_distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3527b9de",
   "metadata": {},
   "outputs": [],
   "source": [
    "from facenet_models import FacenetModel\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.patches import Rectangle\n",
    "from matplotlib.pyplot import text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad12a468",
   "metadata": {},
   "outputs": [],
   "source": [
    "def label_faces(image_data):\n",
    "    \"\"\"\n",
    "    Displays an image with boxes around people's faces and labels them with names.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    image_data : numpy.ndarray, shape-(R, C, 3) (RGB is the last dimension)\n",
    "        Pixel information for the image.\n",
    "    \"\"\"\n",
    "    \n",
    "    # this will download the pretrained weights (if they haven't already been fetched)\n",
    "    # which should take just a few seconds\n",
    "    model = FacenetModel()\n",
    "\n",
    "    # detect all faces in an image\n",
    "    # returns a tuple of (boxes, probabilities, landmarks)\n",
    "    boxes, probabilities, _ = model.detect(image_data)\n",
    "\n",
    "    # producing a face descriptor for each face\n",
    "    # returns a (N, 512) array, where N is the number of boxes\n",
    "    # and each descriptor vector is 512-dimensional\n",
    "    fig, ax = plt.subplots()\n",
    "    ax.imshow(image_data)\n",
    "    if (boxes is None):\n",
    "        x = image_data.shape[1] // 2\n",
    "        y = image_data.shape[0] // 2\n",
    "        ax.text(x, y,\n",
    "                \"GIMME A FACE\",\n",
    "                size=50,\n",
    "                va=\"center\",\n",
    "                ha=\"center\",\n",
    "                bbox=dict(boxstyle=\"round\", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))\n",
    "        return\n",
    "    descriptors = model.compute_descriptors(image_data, boxes)\n",
    "\n",
    "    names = []\n",
    "    for d in descriptors:\n",
    "        names.append(query_database(d).capitalize())\n",
    "\n",
    "    i = 0\n",
    "    for box, prob in zip(boxes, probabilities):\n",
    "        # draw the box on the screen\n",
    "        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color=\"red\"))\n",
    "        # add names to the box\n",
    "        ax.text(*box[:2],\n",
    "                names[i],\n",
    "                size=12,\n",
    "                va=\"center\",\n",
    "                bbox=dict(boxstyle=\"round\", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))\n",
    "        i += 1\n",
    "        \n",
    "    return names"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
