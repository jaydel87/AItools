import imageOperations as im
import imageFeatures as imf
import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBRegressor
import matplotlib.pyplot as plt

image = im.newImage()
image.numberOfCategories = 2
image.categoryNames = ["Category"]

trainingImage = im.readImage(image)
rows = trainingImage.shape[0]
cols = trainingImage.shape[1]

trainingImageNorm = trainingImage / float(np.max(trainingImage))
cv2.namedWindow(winname=image.imagePath)

added_image = trainingImageNorm[:, :, 0]

pixelID = np.arange(0, rows*cols)

features_df = pd.DataFrame(data=pixelID, columns=["PixelID"])
features_df["PixelValue"] = trainingImage[:, :, 0].flatten()

features = imf.newFeature()
imf.get_all_features(features, trainingImage[:, :, 0])

print(len(features.featureNames), len(features.features))

for i in range(len(features.featureNames)):
    features_df[features.featureNames[i]] = features.features[i]

print(features_df.describe())

trainingPixels = np.zeros([rows, cols])

for n in range(image.numberOfCategories):

    categories = im.getCategories()
    cv2.setMouseCallback(image.imagePath, categories.record_pixels)

    im.displayImage(image, trainingImageNorm)

    for i in range(len(categories.xlist)):
        trainingPixels[categories.ylist[i], categories.xlist[i]] = n+1

features_df['Category'] = trainingPixels.flatten()

added_image = cv2.addWeighted(added_image, 1.0, trainingPixels, 1.0, 0)

im.displayImage(image, added_image)

X_pred = features_df.copy()
X_pred.drop(['Category'], axis=1, inplace=True)

idx = features_df['Category'].to_numpy().nonzero()
print(idx)

training_df = features_df.iloc[idx]
print(training_df.describe())
y = training_df.Category
X = training_df.copy()
X.drop(['Category'], axis=1, inplace=True)

model = GradientBoostingClassifier(random_state=1)

model.fit(X, y)
prediction = model.predict(X_pred)

plt.imshow(np.array(prediction).reshape((rows, cols)))
plt.show()