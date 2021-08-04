import imageOperations as im
import imageFeatures as imf
import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import random as rnd

image = im.newImage()
image.numberOfCategories = 4
image.categoryNames = ["Category"]

trainingImage = im.readImage(image)
rows = trainingImage.shape[0]
cols = trainingImage.shape[1]
channels = image.imageChannels

trainingImageNorm = trainingImage / float(np.max(trainingImage))
cv2.namedWindow(image.imagePath, cv2.WINDOW_GUI_EXPANDED)

added_image = trainingImage

pixelID = np.arange(0, rows*cols)

features_df = pd.DataFrame(data=pixelID, columns=["PixelID"])

if channels == 1:
    features_df["GreyValue"] = trainingImage.flatten()
    im.channelNames = [""]
    dispImage = trainingImage
elif channels == 3:
    features_df["BlueValue"] = trainingImage[:, :, 0].flatten()
    features_df["GreenValue"] = trainingImage[:, :, 1].flatten()
    features_df["RedValue"] = trainingImage[:, :, 2].flatten()
    im.channelNames = ["Blue", "Green", "Red"]
    dispImage = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2RGB)
else:
    print("Strange number of channels detected!")

featureNames = []

for i in range(channels):
    features = imf.newFeature()
    imf.get_all_features(features, trainingImage, channels, i)
    featureNames = [x + im.channelNames[i] for x in features.featureNames]

    for j in range(len(featureNames)):
        features_df[featureNames[j]] = features.features[j]

print(features_df.describe())

trainingPixels = np.zeros([rows, cols])

for n in range(image.numberOfCategories):
    r = rnd.randint(0, 255)
    g = rnd.randint(0, 255)
    b = rnd.randint(0, 255)

    colour = (r, g, b)

    categories = im.getCategories(dispImage, colour)
    cv2.setMouseCallback(image.imagePath, categories.record_pixels)

    im.displayImage(image, dispImage)

    for i in range(len(categories.xlist)):
        trainingPixels[categories.ylist[i], categories.xlist[i]] = n+1

features_df['Category'] = trainingPixels.flatten()

#added_image = cv2.addWeighted(added_image, 1.0, trainingPixels, 1.0, 0)

im.displayImage(image, dispImage)

X_pred = features_df.copy()
X_pred.drop(['Category'], axis=1, inplace=True)

idx = features_df['Category'].to_numpy().nonzero()
print(idx)

training_df = features_df.iloc[idx]
print(training_df.describe())
y = training_df.Category
X = training_df.copy()
X.drop(['Category'], axis=1, inplace=True)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#model = GradientBoostingClassifier(random_state=1, n_estimators=5000, learning_rate=0.05)
#model = RandomForestClassifier(random_state=1, n_estimators=100)
model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05)


model.fit(train_X, train_y)
pred = model.predict(train_X)
val_pred = model.predict(val_X)
mae = mean_absolute_error(val_pred, val_y)
print(mae)
prediction = model.predict(X_pred)

plt.imshow(prediction.reshape((rows, cols)), cmap='Greys')
plt.show()

