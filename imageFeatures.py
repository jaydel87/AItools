import skimage.filters
import skimage.feature
import skimage.morphology


class newFeature:

    def __init__(self):
        self.featureNames = []
        self.features = []


def gaussian(image):
    return skimage.filters.gaussian(image)


def mean(image):
    return skimage.filters.rank.mean(image)


def bilateral_mean(image):
    return skimage.filters.rank.mean_bilateral(image)


def median(image):
    return skimage.filters.rank.median(image)


def opening(image):
    return skimage.morphology.opening(image)


def closing(image):
    return skimage.morphology.closing(image)


def dilation(image):
    return skimage.morphology.dilation(image)


def erosion(image):
    return skimage.morphology.erosion(image)


def edges(image):
    return skimage.filters.sobel(image)


def gradient(image):
    return skimage.filters.rank.gradient(image)


def laplace(image):
    return skimage.filters.laplace(image)


def get_all_features(self, image):

    self.features.append(gaussian(image).flatten())
    #self.features.append(mean(image).flatten())
    #self.features.append(bilateral_mean(image).flatten())
    self.features.append(median(image).flatten())
    self.features.append(opening(image).flatten())
    self.features.append(closing(image).flatten())
    self.features.append(dilation(image).flatten())
    self.features.append(erosion(image).flatten())
    self.features.append(edges(image).flatten())
    #self.features.append(gradient(image).flatten())
    self.features.append(laplace(image).flatten())

    #self.featureNames = ["Gaussian", "Mean", "BiMean", "Med", "Open", "Close", "Dilate", "Erode", "Edges", "Grad", "Laplace"]
    self.featureNames = ["Gaussian", "Med", "Open", "Close", "Dilate", "Erode", "Edges",
                         "Laplace"]



