import imageOperations as im
import cv2
import numpy as np

image = im.newImage()
image.numberOfFeatures = 2

trainingImage = im.readImage(image)
rows = trainingImage.shape[0]
cols = trainingImage.shape[1]

featureLists = np.zeros([image.numberOfFeatures, rows*cols])

trainingImageNorm = trainingImage / float(np.max(trainingImage))
cv2.namedWindow(winname=image.imagePath)

added_image = trainingImageNorm[:, :, 0]

for n in range(image.numberOfFeatures):

    feature = im.features()
    cv2.setMouseCallback(image.imagePath, feature.record_pixels)

    im.displayImage(image, trainingImageNorm)
    trainingPixels = np.zeros([rows, cols])

    for i in range(len(feature.xlist)):
        trainingPixels[feature.ylist[i], feature.xlist[i]] = (n+1)*0.5

    added_image = cv2.addWeighted(added_image, 1.0, trainingPixels, 1.0, 0)
im.displayImage(image, added_image)