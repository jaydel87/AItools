import skimage.filters
import skimage.feature
import skimage.morphology
import skimage.color
import skimage.restoration
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class newFeatureSet:

    def __init__(self):
        self.featureNames = []
        self.features = []
        self.selectedFeatures = ["Mean", "Med", "Min", "Max", "Variance"]

def laplace(self, image):
    self.featureNames.append("Laplace")
    return skimage.filters.laplace(image)

def mean(self, patch):
    self.featureNames.append("Mean")
    #print("mean: "+str(np.mean(patch)))
    return np.mean(patch)

def median(self, patch):
    self.featureNames.append("Med")
    #print("median: "+str(np.median(patch)))
    return np.median(patch)

def mode(self, patch):
    self.featureNames.append("Mode")
    return stats.mode(patch)[0]

def max(self, patch):
    self.featureNames.append("Max")
    #print("max: "+str(np.amax(patch)))
    return np.amax(patch)

def min(self, patch):
    self.featureNames.append("Min")
    #print("min: "+str(np.amin(patch)))
    return np.amin(patch)

def variance(self, patch):
    self.featureNames.append("Variance")
    #print("variance: "+str(np.var(patch)))
    return np.var(patch)

def histogram(patch):
    return np.histogram(patch, density=True)

def entropy(self, patch):
    self.featureNames.append("Entropy")
    print(histogram(patch)[0])
    return -np.sum([x * np.log2(x) for x in histogram(patch)[0]])





keyDict = {
    "Mean": mean,
    "Med": median,
    "Mode": mode,
    "Max": max,
    "Min": min,
    "Variance": variance,
    "Entropy": entropy,
}

def get_all_features(self, image):

    self.features = []

    for key in keyDict:
        #print(key)
        self.features.append(keyDict[key](self, image).flatten())

    return keyDict.keys()


def get_selected_features(self, image, channel, patches):

    #self.features = []
    noPatches = np.amax(patches)

    # print(channel, image[:, :, channel])
    image = image[:, :, channel]


    for key in self.selectedFeatures:
        segmentFeatures = np.zeros(noPatches)
        for patch in range(1, noPatches+1):
            mask = (patches == patch)
            maskedImage = list(image[mask])

            feature = keyDict[key](self, maskedImage)
            segmentFeatures[patch-1] = feature
            #print(key, feature)

        #print(key, segmentFeatures)
        self.features.append(segmentFeatures)

    #print(len(self.features))

    return self.selectedFeatures






