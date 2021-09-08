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


def max(self, image):
    self.featureNames.append("Max")
    return skimage.filters.rank.maximum(image, skimage.morphology.square(3))


def min(self, image):
    self.featureNames.append("Min")
    return skimage.filters.rank.minimum(image, skimage.morphology.square(3))


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

def hog(self, image):
    self.featureNames.append("HistGrad")
    hog, hog_image = skimage.feature.hog(image, visualize=True)

    #plt.imshow(hog_image)
    #plt.show()

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


def entropy(self, image):
    self.featureNames.append("Entropy")
    return skimage.filters.rank.entropy(image, skimage.morphology.square(3))

def get_all_features(self, image, channels, channel):

    self.features = []

    if channels > 1:
        image = image[:, :, channel]

    self.features.append(gaussian(self, image).flatten())
    self.features.append(mean(self, image).flatten())
    #self.features.append(bilateral_mean(self, image).flatten())
    self.features.append(median(self, image).flatten())
    self.features.append(opening(self, image).flatten())
    self.features.append(closing(self, image).flatten())
    self.features.append(dilation(self, image).flatten())
    self.features.append(erosion(self, image).flatten())
    self.features.append(edges(self, image).flatten())
    self.features.append(gradient(self, image).flatten())
    self.features.append(laplace(self, image).flatten())
    #self.features.append(hog(self, image).flatten())
    self.features.append(dog(self, image).flatten())
    self.features.append(gabor(self, image).flatten())
    self.features.append(lbp(self, image).flatten())
    self.features.append(hessian(self, image).flatten())
    self.features.append(wavelet_denoising(self, image).flatten())
    self.features.append(entropy(self, image).flatten())




