import imageOperations as im
import imageFeatures as imf
import imageAI as ai
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from sklearn import tree
import matplotlib.pyplot as plt
import random as rnd
import time
from PIL import Image

model, features = ai.load_model("RandomForest")
model_boost, features = ai.load_model("GradientBoost")

print(features)

image = im.newImage()

# Read and image and get its properties
predictionImage = im.readImage(image)

imName = image.imagePath.rsplit(".", 1)[0]
print(imName)

rows = predictionImage.shape[0]
cols = predictionImage.shape[1]
channels = image.imageChannels

# Name the pop-up window: Necessary for interactivity
cv2.namedWindow(image.imagePath, cv2.WINDOW_GUI_EXPANDED)

# Gives each pixel in the image a unique ID
pixelID = np.arange(0, rows*cols)

# Initialise a dataframe for the image information with a column "PixelID"
features_df = pd.DataFrame(data=pixelID, columns=["PixelID"])

# Add a column(s) for each pixel value. Data is stored in 1D vector format.
# For greyscale images: one column for Grey value
# For RGB images: one column for each value of R, G and B
# TO DO: HIDE THIS IN imageFeatures.py
if channels == 1:
    features_df["GreyValue"] = predictionImage.flatten()
    im.channelNames = [""]
    dispImage = predictionImage
elif channels == 3:
    features_df["BlueValue"] = predictionImage[:, :, 0].flatten()
    features_df["GreenValue"] = predictionImage[:, :, 1].flatten()
    features_df["RedValue"] = predictionImage[:, :, 2].flatten()
    im.channelNames = ["Blue", "Green", "Red"]
    dispImage = cv2.cvtColor(predictionImage, cv2.COLOR_BGR2RGB)
else:
    print("Strange number of channels detected!")

featureNames = []

# Calculate the features and add a column for each feature and each filter size in the dataframe.
# As before, data is stored in 1D vector format
for i in range(channels):
    features = imf.newFeatureSet()
    #features.selectedFeatures = features['Names']
    features.kernelSizes = [3, 7]
    image_features, feature_sizes = imf.get_selected_features(features, predictionImage, channels, i, features.kernelSizes)
    featureNames = [x + im.channelNames[i] for x in features.featureNames]

    for j in range(len(featureNames)):
        features_df[featureNames[j]] = features.features[j]

# Prints statistics from dataframe
#print(features_df.describe())

im.displayImage(image, dispImage)

# Copy the initial dataframe for later applying the trained model
# Remove columns which should not be included for the prediction
X_pred = features_df.copy()
X_pred.drop(['PixelID'], axis=1, inplace=True)

t0 = time.time()
prediction = ai.predict(model, X_pred)
prob_cat, prob_all = ai.model_probability(model, X_pred, prediction)
t1 = time.time()
print("Time (s) for random forest: "+str(t1-t0))

t0 = time.time()
prediction_boost = ai.predict(model_boost, X_pred)
prob_boost_cat, prob_boost_all = ai.model_probability(model_boost, X_pred, prediction_boost)
t1 = time.time()
print("Time (s) for gradient boosting: "+str(t1-t0))

print(np.mean(prob_all), np.mean(prob_boost_all))

for i in range(len(X_pred.columns)):
    print(X_pred.columns[i], model_boost.feature_importances_[i])

#ai.save_model(model, "RandomForest1", image_features, feature_sizes)
#ai.save_model(model_boost, "GradientBoost1", image_features, feature_sizes)
#
#print(ai.load_model("RandomForest1"))

fig1 = plt.figure()
prediction_arr = prediction.reshape((rows, cols))
plt.imshow(prediction.reshape((rows, cols)), cmap='Greys')
prediction_im = Image.fromarray(prediction_arr)
prediction_im.save(imName + "_bin.tif")


fig2 = plt.figure()
prediction_arr = prediction_boost.reshape((rows, cols))
plt.imshow(prediction_boost.reshape((rows, cols)), cmap='Greys')
prediction_im = Image.fromarray(prediction_arr)
prediction_im.save(imName + "_GB_bin.tif")

fig3 = plt.figure()
plt.imshow(prob_boost_all.reshape((rows, cols)), cmap='Greys')

# fig3 = plt.figure()
# plt.imshow(prob_all.reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)
#
# fig4 = plt.figure()
# plt.imshow(prob_boost_all.reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

plt.show()

