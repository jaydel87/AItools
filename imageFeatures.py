import skimage.filters
import skimage.feature
import skimage.morphology
import skimage.color
import skimage.restoration
import matplotlib.pyplot as plt


class newFeature:

    def __init__(self):
        self.featureNames = []
        self.features = []
        self.kernelSizes = [3]
        self.selectedFeatures = ["Gaussian", "Mean", "Med", "Min", "Max", "Open", "Close", "Edge", "Wavelet"]


def gaussian(self, image, size):
    self.featureNames.append("Gaussian" + str(size))
    return skimage.filters.gaussian(image, sigma=1, truncate=size)


def mean(self, image, size):
    self.featureNames.append("Mean" + str(size))
    return skimage.filters.rank.mean(image, skimage.morphology.square(size))


def bilateral_mean(self, image, size):
    self.featureNames.append('BiMean' + str(size))
    return skimage.filters.rank.mean_bilateral(image, skimage.morphology.square(size))


def median(self, image, size):
    self.featureNames.append("Med" + str(size))
    return skimage.filters.rank.median(image, skimage.morphology.square(size))


def max(self, image, size):
    self.featureNames.append("Max" + str(size))
    return skimage.filters.rank.maximum(image, skimage.morphology.square(size))


def min(self, image, size):
    self.featureNames.append("Min" + str(size))
    return skimage.filters.rank.minimum(image, skimage.morphology.square(size))


def opening(self, image, size):
    self.featureNames.append("Open" + str(size))
    return skimage.morphology.opening(image, skimage.morphology.square(size))


def closing(self, image, size):
    self.featureNames.append("Close" + str(size))
    return skimage.morphology.closing(image, skimage.morphology.square(size))


def dilation(self, image, size):
    self.featureNames.append("Dilate" + str(size))
    return skimage.morphology.dilation(image, skimage.morphology.square(size))


def erosion(self, image, size):
    self.featureNames.append("Erode" + str(size))
    return skimage.morphology.erosion(image, skimage.morphology.square(size))


def gradient(self, image, size):
    self.featureNames.append("Grad" + str(size))
    return skimage.filters.rank.gradient(image, skimage.morphology.square(size))


def entropy(self, image, size):
    self.featureNames.append("Entropy" + str(size))
    return skimage.filters.rank.entropy(image, skimage.morphology.square(size))


def edges(self, image):
    self.featureNames.append("Edge")
    return skimage.filters.sobel(image)


def laplace(self, image):
    self.featureNames.append("Laplace")
    return skimage.filters.laplace(image)


def hog(self, image):
    self.featureNames.append("HistGrad")
    hog, hog_image = skimage.feature.hog(image, visualize=True)

    # plt.imshow(hog_image)
    # plt.show()

    return skimage.feature.hog(image)


def dog(self, image):
    self.featureNames.append("DiffGauss")
    return skimage.filters.difference_of_gaussians(image, 0.1)


def gabor(self, image):
    self.featureNames.append("Gabor")
    return skimage.filters.gabor(image, frequency=0.5)[0]


def lbp(self, image):
    self.featureNames.append("LBP")
    grey_image = skimage.color.rgb2gray(image)
    return skimage.feature.local_binary_pattern(grey_image, 3, 10)


def hessian(self, image):
    self.featureNames.append("Hessian")
    return skimage.filters.hessian(image, sigmas=range(1, 10, 2))


def wavelet_denoising(self, image):
    self.featureNames.append("Wavelet")
    return skimage.restoration.denoise_wavelet(image)


keyDict = {
    "Gaussian": gaussian,
    "Mean": mean,
    #"BiMean": bilateral_mean,
    "Med": median,
    "Max": max,
    "Min": min,
    "Open": opening,
    "Close": closing,
    "Dilate": dilation,
    "Erode": erosion,
    "Grad": gradient,
    "Entropy": entropy,
    "Edge": edges,
    "Laplace": laplace,
    #"HistGrad": hog,
    "DiffGauss": dog,
    "Gabor": gabor,
    "LBP": lbp,
    "Hessian": hessian,
    "Wavelet": wavelet_denoising
}

kernel_based_features = ["Gaussian", "Mean", "BiMean", "Med", "Max", "Min", "Open", "Close", "Dilate", "Erode", "Grad",
                         "Entropy"]


def get_all_features(self, image, channels, channel, sizes):

    self.features = []

    if channels > 1:
        image = image[:, :, channel]

    for key in keyDict:
        print(key)
        if key in kernel_based_features:
            for size in sizes:
                self.features.append(keyDict[key](self, image, size).flatten())
        else:
            self.features.append(keyDict[key](self, image).flatten())


def get_selected_features(self, image, channels, channel, sizes):

    self.features = []

    if channels > 1:
        image = image[:, :, channel]

    for key in self.selectedFeatures:
        print(key)
        if key in kernel_based_features:
            for size in sizes:
                self.features.append(keyDict[key](self, image, size).flatten())
        else:
            self.features.append(keyDict[key](self, image).flatten())







