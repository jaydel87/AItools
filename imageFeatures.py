import skimage.filters
import skimage.feature
import skimage.morphology
import matplotlib.pyplot as plt


class newFeature:

    def __init__(self):
        self.featureNames = []
        self.features = []


def gaussian(self, image):
    self.featureNames.append("Gaussian")
    return skimage.filters.gaussian(image, sigma=1, truncate=3)


def mean(self, image):
    self.featureNames.append("Mean")
    return skimage.filters.rank.mean(image, skimage.morphology.square(3))


def bilateral_mean(self, image):
    self.featureNames.append('BiMean')
    return skimage.filters.rank.mean_bilateral(image, skimage.morphology.square(5))


def median(self, image):
    self.featureNames.append("Med")
    return skimage.filters.rank.median(image, skimage.morphology.square(3))


def opening(self, image):
    self.featureNames.append("Open")
    return skimage.morphology.opening(image, skimage.morphology.square(3))


def closing(self, image):
    self.featureNames.append("Close")
    return skimage.morphology.closing(image, skimage.morphology.square(3))


def dilation(self, image):
    self.featureNames.append("Dilate")
    return skimage.morphology.dilation(image, skimage.morphology.square(3))


def erosion(self, image):
    self.featureNames.append("Erode")
    return skimage.morphology.erosion(image, skimage.morphology.square(3))


def edges(self, image):
    self.featureNames.append("Edge")
    return skimage.filters.sobel(image)


def gradient(self, image):
    self.featureNames.append("Grad")
    return skimage.filters.rank.gradient(image, skimage.morphology.square(3))


def laplace(self, image):
    self.featureNames.append("Laplace")
    return skimage.filters.laplace(image)


def get_all_features(self, image, channels, channel):

    self.features = []

    if channels > 1:
        image = image[:, :, channel]

    #self.features.append(gaussian(self, image).flatten())
    self.features.append(mean(self, image).flatten())
    #self.features.append(bilateral_mean(self, image).flatten())
    # self.features.append(median(self, image).flatten())
    # self.features.append(opening(self, image).flatten())
    # self.features.append(closing(self, image).flatten())
    # self.features.append(dilation(self, image).flatten())
    # self.features.append(erosion(self, image).flatten())
    self.features.append(edges(self, image).flatten())
    # self.features.append(gradient(self, image).flatten())
    # self.features.append(laplace(self, image).flatten())





