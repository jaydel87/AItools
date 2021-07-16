import imageOperations as im
import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

image = im.newImage()
image.numberOfFeatures = 2
image.featureNames = ["Background"]#, "Cell"]

trainingImage = im.readImage(image)
rows = trainingImage.shape[0]
cols = trainingImage.shape[1]

trainingImageNorm = trainingImage / float(np.max(trainingImage))
cv2.namedWindow(winname=image.imagePath)

added_image = trainingImageNorm[:, :, 0]

pixelID = np.arange(0, rows*cols)

features_df = pd.DataFrame(data=pixelID, columns=["PixelID"])
features_df["PixelValue"] = trainingImage[:, :, 0].flatten()

trainingPixels = np.zeros([rows, cols])

for n in range(image.numberOfFeatures):

    feature = im.features()
    cv2.setMouseCallback(image.imagePath, feature.record_pixels)

    featureName = image.featureNames[0]


    im.displayImage(image, trainingImageNorm)

    for i in range(len(feature.xlist)):
        trainingPixels[feature.ylist[i], feature.xlist[i]] = n+1

features_df[featureName] = trainingPixels.flatten()

added_image = cv2.addWeighted(added_image, 1.0, trainingPixels, 1.0, 0)

im.displayImage(image, added_image)

idx = np.argwhere(features_df.Background.values != 0)
y = features_df.Background.values[idx]
X = features_df.PixelValue.values[idx]

model = DecisionTreeRegressor(random_state=1)
model.fit(np.array(X).reshape(-1, 1), y)
prediction = model.predict(np.array(features_df.PixelValue).reshape(-1, 1))

prediction_int = [round(x) for x in prediction]

plt.imshow(np.array(prediction).reshape((rows, cols)))
plt.show()