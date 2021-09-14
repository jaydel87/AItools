import imageOperations as im
import imageFeatures as imf
#import imageAI as ai
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
import time

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
    features.kernelSizes = [3, 5, 7, 9]
    imf.get_selected_features(features, trainingImage, channels, i, features.kernelSizes)
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
X_pred.drop(['PixelID'], axis=1, inplace=True)

idx = features_df['Category'].to_numpy().nonzero()
#print(idx)

training_df = features_df.iloc[idx]
print(training_df.describe())
y = training_df.Category
X = training_df.copy()
X.drop(['Category'], axis=1, inplace=True)
X.drop(['PixelID'], axis=1, inplace=True)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

estimators = 200
depth = 5
rate = 0.1
features = len(X_pred.columns) #int(0.5*len(X_pred.columns))
L1reg = 0.1

#model = GradientBoostingClassifier(random_state=1, n_estimators=5000, learning_rate=0.05)
model = RandomForestClassifier(random_state=1, n_estimators=estimators, max_depth=depth, max_features=features)
model_boost = xgb.XGBClassifier(n_estimators=estimators, max_depth=depth, learning_rate=rate, reg_alpha=L1reg)

t0 = time.time()
model.fit(train_X, train_y)
pred = model.predict(train_X)
val_pred = model.predict(val_X)
mae = mean_absolute_error(val_pred, val_y)
print(mae)
prediction = model.predict(X_pred)
prob = model.predict_proba(X_pred)
print(prob)
prob1 = np.transpose(prob, [1, 0])
t1 = time.time()
print("Time (s) for random forest: "+str(t1-t0))
print(prob1)

t0 = time.time()
model_boost.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(val_X, val_y)])
pred_boost = model_boost.predict(train_X)
val_pred_boost = model_boost.predict(val_X)
mae_boost = mean_absolute_error(val_pred_boost, val_y)
print(mae_boost)
prediction_boost = model_boost.predict(X_pred)
proba = model_boost.predict_proba(X_pred)
proba1 = np.transpose(proba, [1, 0])
t1 = time.time()
print("Time (s) for gradient boosting: "+str(t1-t0))

for i in range(len(X_pred.columns)):
    print(X_pred.columns[i], model_boost.feature_importances_[i])
# importances = model.feature_importances_
# print(X_pred.columns)
# print(importances)


fig1 = plt.figure()
plt.imshow(prediction.reshape((rows, cols)), cmap='Greys')

fig2 = plt.figure()
plt.imshow(prediction_boost.reshape((rows, cols)), cmap='Greys')

fig3 = plt.figure()
plt.imshow(prob1[0].reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

fig4 = plt.figure()
plt.imshow(prob1[1].reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

fig5 = plt.figure()
plt.imshow(prob1[2].reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

fig6 = plt.figure()
plt.imshow(prob1[3].reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

plt.show()

