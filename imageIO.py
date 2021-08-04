import easygui
import skimage.io as io

class newImage:

    def __init__(self):

        self.imagePath = "default image path"
        self.displayTitle = "Display"
        self.numberOfCategories = 0
        self.categoryNames = []
        self.paintBrushSize = 1
        self.paintBrushColour = 0


def readImage(self):

    self.imagePath = easygui.fileopenbox("Please select an image")
    im = io.imread(self.imagePath, plugin='pil')

    return im


def displayImage(self, im):

    io.imshow(self.imagePath, im)


class getCategories():

    def __init__(self):
        self.xlist = []
        self.ylist = []
        self.lclick = False

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






