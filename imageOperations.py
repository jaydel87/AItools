import easygui
import cv2
import numpy as np
from skimage.io import imread
from skimage.segmentation import mark_boundaries

class newImage:

    def __init__(self):

        self.imagePath = "default image path"
        self.imageChannels = 1
        self.channelNames = []
        self.displayTitle = "Display"
        self.numberOfCategories = 0
        self.categoryNames = []
        self.paintBrushSize = 1
        self.paintBrushColour = 0
        self.useSavedPixels = 0
        self.noDraw = 0
        self.maskPath = "default image path"


def readImage(self):

    self.imagePath = easygui.fileopenbox("Please select an image")
    #im = cv2.imread(self.imagePath)
    im = imread(self.imagePath, plugin='pil')
    self.imageChannels = 1 if len(im.shape) < 3 else im.shape[-1]

    return im


def displayImage(self, im, patches):

    cv2.imshow(self.imagePath, im)
    cv2.imshow(self.imagePath, mark_boundaries(im, patches))
    cv2.waitKey(0)


class getCategories:

    def __init__(self, image, colour):
        self.xlist = []
        self.ylist = []
        self.lclick = False
        self.image = image
        self.colour = colour

    def record_pixels(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.lclick = True
            self.xlist.append(x)
            self.ylist.append(y)

        if event == cv2.EVENT_LBUTTONUP:
            self.lclick = False

        if self.lclick == True and event == cv2.EVENT_MOUSEMOVE:
            self.xlist.append(x)
            self.ylist.append(y)

            cv2.line(self.image, (self.xlist[-2], self.ylist[-2]), (self.xlist[-1], self.ylist[-1]), self.colour, 1)







